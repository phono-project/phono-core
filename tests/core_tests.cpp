#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "context/context.hpp"
#include "core/config.hpp"
#include "core/tokenizer.hpp"

namespace {

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("cannot create test file");
    out << contents;
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

phono::core::ModelPackageConfig test_config() {
    phono::core::ModelPackageConfig cfg;
    cfg.pre_model.max_seqlen = 16;
    cfg.pre_model.mhsa_layers = 1;
    cfg.pre_model.mhsa_heads = 1;
    cfg.pre_model.attn_dim = 4;
    cfg.post_model.mhca_heads = 1;
    cfg.post_model.mhca_attn_dim = 4;
    cfg.runtime.batch_size = 2;
    return cfg;
}

void test_tokenizer_normalizes_and_skips_unknown() {
    const auto dir = std::filesystem::temp_directory_path() / "phono_core_tokenizer_test";
    std::filesystem::create_directories(dir);
    write_file(dir / "chinese.txt", "\xE5\x8F\xB0\nA\n");
    write_file(dir / "context.txt", "\xE5\x8F\xB0\nA\nknown\n");
    write_file(dir / "pinyin.txt", "ni\n");

    phono::core::Tokenizer tokenizer(
        (dir / "chinese.txt").string(), (dir / "context.txt").string(),
        (dir / "pinyin.txt").string(), {"bos_token"});

    // U+81FA (traditional Taiwan character) -> U+53F0 and full-width A -> A.
    const auto ids = tokenizer.encode_context("\xE8\x87\xBA\xEF\xBC\xA1X");
    check(ids.size() == 2, "normalization should produce two known ids");
    check(ids[0] == 0, "traditional character should simplify");
    check(ids[1] == 1, "full-width character should be NFKC normalized");
    check(tokenizer.encode_context("X").empty(), "unknown context ids should be skipped");
    check(tokenizer.chinese_id_to_context_id(0) == 0, "Chinese/context ids should map");
    check(tokenizer.chinese_id_to_context_id(99) == -1, "unknown Chinese id should fail");

    std::filesystem::remove_all(dir);
}

void test_cache_slice_operations() {
    auto cache = phono::context::make_zero_persistent_tensor({1, 2, 2, 8, 1, 1});
    for (int32_t kv = 0; kv < 2; ++kv) {
        for (int32_t batch = 0; batch < 2; ++batch) {
            for (int32_t pos = 0; pos < 8; ++pos) {
                cache.data()[(kv * 2 + batch) * 8 + pos] =
                    static_cast<float>(100 * kv + 10 * batch + pos);
            }
        }
    }
    cache.shift_batch_tokens(0, 2, 3);
    check(cache.data()[0] == 0.0f, "BOS is protected");
    check(cache.data()[1] == 3.0f, "shift should copy source position 3");
    check(cache.data()[2] == 4.0f, "shift should preserve order");
    check(cache.data()[3] == 5.0f, "shift should preserve retained tokens");

    cache.copy_batch_slice_to_all(0, 1, 3);
    check(cache.data()[8 + 1] == 3.0f, "slice should copy to beam one");
    check(cache.data()[8 + 2] == 4.0f, "slice should copy to beam one");
    check(cache.data()[8 + 3] == 5.0f, "slice should copy to beam one");

    cache.reorder_batches({1, 0});
    check(cache.data()[1] == 3.0f, "reorder should preserve selected beam");
    check(cache.data()[8 + 1] == 3.0f, "reorder should preserve selected beam");
}

void test_context_auto_matching() {
    const auto cfg = test_config();
    const nlohmann::json options = {
        {"beam_size", 2}, {"N", 2}, {"T", 2}, {"max_context_length", 8},
    };
    phono::context::ContextManager manager(cfg, 2, options);
    auto& slot = manager.get_context_by_id(0);
    const int32_t ids[] = {1, 2, 3, 4};
    slot.append_context_ids(ids, 4);
    slot.set_history_seqlen(5);
    slot.set_current_position(5);
    const int32_t cache_max = slot.self_kv.max_seqlen();
    const int64_t token_stride = slot.self_kv.token_stride_elements();
    const int64_t batch_stride = slot.self_kv.batch_stride_elements();
    for (int32_t kv = 0; kv < 2; ++kv) {
        for (int32_t batch = 0; batch < 2; ++batch) {
            for (int32_t pos = 0; pos < cache_max; ++pos) {
                const int64_t offset = (kv * 2 + batch) * batch_stride + pos * token_stride;
                for (int64_t d = 0; d < token_stride; ++d) {
                    slot.self_kv.data()[offset + d] = static_cast<float>(pos);
                }
            }
        }
    }

    slot.shift_cached_tokens(2, 2);
    check(slot.self_kv.data()[token_stride] == 3.0f,
          "direct cache shift should preserve first match");
    slot.self_kv.zero_();
    for (int32_t kv = 0; kv < 2; ++kv) {
        for (int32_t batch = 0; batch < 2; ++batch) {
            for (int32_t pos = 0; pos < cache_max; ++pos) {
                const int64_t offset = (kv * 2 + batch) * batch_stride + pos * token_stride;
                for (int64_t d = 0; d < token_stride; ++d) {
                    slot.self_kv.data()[offset + d] = static_cast<float>(pos);
                }
            }
        }
    }
    auto* shifted = manager.get_context_auto({3, 4, 5});
    check(shifted == &slot, "middle match should reuse the existing slot");
    check(shifted->context_ids_len() == 2, "middle match should retain matched ids");
    check(shifted->context_ids()[0] == 3, "matched id should move after BOS");
    check(shifted->context_ids()[1] == 4, "matched id should move after BOS");
    check(shifted->history_seqlen() == 3, "history should include BOS");
    check(shifted->current_position() == 3, "current length should include BOS");
    check(shifted->self_kv.data()[token_stride] == 3.0f,
          "cache shift should preserve first match");
    check(shifted->self_kv.data()[2 * token_stride] == 4.0f,
          "cache shift should preserve second match");
    check(shifted->self_kv.data()[batch_stride + token_stride] == 3.0f,
          "cache shift should affect every beam");

    auto* truncated = manager.get_context_auto({3});
    check(truncated == &slot, "prefix match should reuse the slot");
    check(truncated->context_ids_len() == 1, "prefix match should truncate ids");
    check(truncated->history_seqlen() == 2, "prefix match should truncate history");
    check(truncated->current_position() == 2, "prefix match should truncate current length");

    auto* divergent = manager.get_context_auto({3, 9});
    check(divergent == &slot, "common prefix should survive a tail edit");
    check(divergent->context_ids_len() == 1, "tail edit should retain the common prefix");
    check(divergent->history_seqlen() == 2, "tail edit should reset cached history length");
}

}  // namespace

int main() {
    test_tokenizer_normalizes_and_skips_unknown();
    test_cache_slice_operations();
    test_context_auto_matching();
    return 0;
}
