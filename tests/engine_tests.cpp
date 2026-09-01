// Model-based session tests. These exercise the real InferenceSession against
// a v2 model package. They require the PHONO_TEST_MODEL_DIR environment
// variable; when it is unset the suite is skipped so CTest still passes on
// machines without a downloaded model.
//
// Covered here:
//   * stateless session driving a context slot through fill + generate
//   * cancellation via a callback + generation-cursor rollback
//   * max_pinyin_length enforcement
//   * max_history_length windowing (eviction keeps BOS as the attention sink)

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "context/context.hpp"
#include "core/config.hpp"
#include "core/tokenizer.hpp"
#include "engine/inference_engine.hpp"

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct CancellationTrigger {
    int max_steps = 0;   // allow this many generation steps, then cancel
    int steps = 0;
};

bool cancel_after_n_steps(void* user_data) {
    auto* trigger = static_cast<CancellationTrigger*>(user_data);
    return trigger->steps++ >= trigger->max_steps;
}

const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

void run(const std::string& package_root) {
    phono::engine::InferenceEngine engine(package_root);
    const auto& cfg = engine.config();
    const auto& tokenizer = engine.tokenizer();

    // A core_config that is valid for any v2 model (pre_max >= 128, post_max
    // >= 32, batch 3) and small enough to exercise windowing quickly.
    phono::core::CoreConfig core;
    core.beam_size = cfg.runtime.batch_size;
    core.max_context_length = cfg.pre_model.max_seqlen - 1;
    core.max_pinyin_length = std::min<int32_t>(4, cfg.post_model.max_seqlen);
    core.max_history_length =
        std::max<int32_t>(1, core.max_context_length - core.max_pinyin_length - 1);
    core.slack_interval = std::max<int32_t>(1, core.max_history_length / 4);

    phono::context::ContextManager manager(cfg, 1, core);
    phono::context::Context* slot = &manager.get_context_by_id(0);
    phono::engine::InferenceSession session(engine, core);

    // --- stateless fill + generate ---
    const std::vector<int32_t> pinyin = tokenizer.encode_pinyin({"ni", "hao"});
    check(!pinyin.empty(), "ni hao should encode to pinyin ids");
    check(slot->context_ids_len() == 0, "a fresh slot must start empty");
    phono::engine::GenerateResult first = session.generate(*slot, pinyin);
    check(first.ok(), "first generate should succeed");
    check(!first.beams.empty(), "first generate should return beams");
    check(static_cast<int32_t>(first.beams.size()) == session.beam_size(),
          "beam count should match the configured beam width");
    check(first.beams[0].decoded == "你好", "expected 'ni hao' to decode to 你好");

    // Reset only invalidates the cursors. Dirty KV data from the previous run
    // must be safely overwritten or masked and produce the same result.
    check(session.reset(*slot) == phono::engine::InferenceError::Ok,
          "reset after generation should succeed");
    first = session.generate(*slot, pinyin);
    check(first.ok() && first.beams[0].decoded == "你好",
          "generation after a dirty-cache reset should remain deterministic");

    const std::vector<int32_t> committed =
        tokenizer.encode_context(first.beams[0].decoded);
    check(session.fill(*slot, committed) == phono::engine::InferenceError::Ok,
          "filling the committed candidate should succeed");
    check(slot->context_ids_len() == static_cast<int32_t>(committed.size()),
          "committed ids should be stored in the context slot");
    check(slot->history_seqlen() == 1 + slot->context_ids_len(),
          "history_seqlen should count BOS + committed ids");

    // A second window with explicit full context: the committed history is
    // reused and only the new suffix is generated against.
    const std::vector<int32_t> context2 = tokenizer.encode_context("你好世界");
    const std::vector<int32_t> pinyin2 = tokenizer.encode_pinyin({"shi", "jie"});
    phono::engine::GenerateResult second =
        session.generate(*slot, pinyin2, context2);
    check(second.ok(), "generate with explicit context should succeed");
    check(second.beams[0].decoded == "世界",
          "expected 'shi jie' after 你好 to decode to 世界");

    // --- cancellation rolls back the generation cursor ---
    const int32_t committed_history_seqlen = slot->history_seqlen();
    const std::vector<int32_t> long_pinyin =
        tokenizer.encode_pinyin({"ni", "hao", "shi", "jie"});
    CancellationTrigger trigger;
    trigger.max_steps = 0;  // cancel on the very first poll (before any token)
    phono::engine::GenerateResult cancelled =
        session.generate(*slot, long_pinyin, {}, cancel_after_n_steps, &trigger);
    check(cancelled.error == phono::engine::InferenceError::Cancelled,
          "cancellation should surface as Cancelled");
    check(cancelled.current_seqlen == cancelled.history_seqlen,
          "cancelled generate must roll back current_seqlen to the committed history");
    check(slot->current_seqlen() == slot->history_seqlen(),
          "context cursors must be rolled back in the slot");
    check(slot->history_seqlen() == committed_history_seqlen,
          "history_seqlen must be unchanged by the cancelled generate");

    // Cancelling mid-generation also rolls the cursor back.
    trigger.steps = 0;
    trigger.max_steps = 1;  // cancel after the first expansion step
    cancelled = session.generate(*slot, long_pinyin, {}, cancel_after_n_steps, &trigger);
    check(cancelled.error == phono::engine::InferenceError::Cancelled,
          "mid-generation cancellation should surface as Cancelled");
    check(cancelled.current_seqlen == cancelled.history_seqlen,
          "mid-generation cancellation must roll back current_seqlen");
    check(slot->current_seqlen() == slot->history_seqlen(),
          "slot cursors must be rolled back after mid-generation cancellation");
    check(slot->history_seqlen() == committed_history_seqlen,
          "history_seqlen must be unchanged by mid-generation cancellation");

    // A retry after cancellation still works from the committed history.
    phono::engine::GenerateResult retry = session.generate(*slot, pinyin);
    check(retry.ok(), "generate after cancellation should succeed");
    check(!retry.beams.empty(), "retry should return beams");
    check(retry.current_seqlen >= retry.history_seqlen,
          "retry cursors should advance consistently from the committed history");

    // --- max_pinyin_length enforcement ---
    const std::vector<int32_t> too_long(core.max_pinyin_length + 1, pinyin.front());
    phono::engine::GenerateResult limited = session.generate(*slot, too_long);
    check(limited.error == phono::engine::InferenceError::PinyinLimitExceeded,
          "pinyin window beyond max_pinyin_length must be rejected");

    // --- max_history_length windowing ---
    session.reset(*slot);
    const int32_t target = core.max_history_length - core.slack_interval;
    std::vector<int32_t> ids;
    for (int32_t i = 0; i < core.max_history_length + 5; ++i) {
        ids.push_back(tokenizer.encode_context("你").front());
    }
    check(session.fill(*slot, ids) == phono::engine::InferenceError::Ok,
          "oversized fill should be windowed instead of failing");
    check(slot->context_ids_len() <= core.max_history_length,
          "fill must keep history within max_history_length");
    check(slot->context_ids_len() >= target,
          "fill windowing must keep at least max_history_length - slack_interval ids");
    check(slot->history_seqlen() == 1 + slot->context_ids_len(),
          "windowed history seqlen must count BOS + retained ids");
}

}  // namespace

int main() {
    const char* model_dir = env_or_null("PHONO_TEST_MODEL_DIR");
    if (model_dir == nullptr) {
        std::printf("engine_tests: PHONO_TEST_MODEL_DIR unset, skipping model-based tests\n");
        return 0;
    }
    try {
        run(model_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "engine_tests: FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("engine_tests: all model-based tests passed\n");
    return 0;
}
