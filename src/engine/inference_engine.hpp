#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

#include "context/context.hpp"
#include "context/kv_cache.hpp"
#include "core/config.hpp"
#include "core/tokenizer.hpp"

namespace phono::engine {

enum class InferenceError {
    Ok = 0,
    InvalidArgument,
    ContextLimitExceeded,
    PinyinLimitExceeded,
    NoCandidates,
    Cancelled,
    ModelError,
};

const char* inference_error_name(InferenceError error);

// Cooperative cancellation callback. `generate` polls this between model
// steps; returning true aborts generation. `user_data` is an opaque context
// pointer passed through unchanged (a thread-safe flag, a signal-handler
// flag, ...). A null callback means cancellation is never requested.
using CancellationFn = bool (*)(void* user_data);

struct BeamResult {
    double score = 0.0;
    std::vector<int32_t> pred_ids;  // Chinese-vocabulary ids.
    std::string decoded;
};

struct GenerateResult {
    InferenceError error = InferenceError::Ok;
    std::vector<BeamResult> beams;
    int32_t current_seqlen = 0;
    int32_t history_seqlen = 0;

    bool ok() const { return error == InferenceError::Ok; }
};

struct PostModelOutput {
    std::vector<float> hidden;
    std::vector<int32_t> candidate_ids;
    std::vector<uint8_t> candidate_mask;
    int32_t batch_size = 0;
    int32_t sequence_length = 0;
    int32_t hidden_dim = 0;
    int32_t candidate_width = 0;
};

struct DecoderModelOutput {
    std::vector<float> logits;
    std::vector<int32_t> candidate_ids;
    int32_t batch_size = 0;
    int32_t sequence_length = 0;
    int32_t projection_size = 0;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& package_root);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    const core::ModelPackageConfig& config() const { return config_; }
    const core::Tokenizer& tokenizer() const { return tokenizer_; }

    // These methods execute the v2.1 exported methods. They return the runtime
    // error instead of throwing so session methods can report a stable code.
    InferenceError run_pre_pass1(const std::vector<int32_t>& input_ids,
                                 context::PersistentTensor& self_kv,
                                 int32_t current_seqlen,
                                 int32_t batch_size) const;

    InferenceError run_pre_pass2(const std::vector<int32_t>& input_ids,
                                 context::PersistentTensor& self_kv,
                                 const std::vector<int32_t>& current_seqlen,
                                 const PostModelOutput& post,
                                 int32_t cross_q_pos_start,
                                 DecoderModelOutput& output) const;

    InferenceError run_post_model(const std::vector<int32_t>& pinyin_ids,
                                  PostModelOutput& output) const;

    int32_t pre_pass1_batch_size() const { return pre_pass1_batch_size_; }
    int32_t pre_pass2_batch_size() const { return pre_pass2_batch_size_; }

private:
    static InferenceError runtime_error_to_status(executorch::runtime::Error error);

    core::ModelPackageConfig config_;
    core::Tokenizer tokenizer_;
    std::unique_ptr<executorch::extension::Module> pre_module_;
    std::unique_ptr<executorch::extension::Module> post_module_;
    int32_t pre_pass1_batch_size_ = 0;
    int32_t pre_pass2_batch_size_ = 0;
    bool pre_pass1_cache_only_ = false;
};

// A stateless inference driver. No context is bound at construction time:
// every call takes the target context::Context explicitly, so a single
// session can drive any number of context slots (see ContextManager) and the
// C-ABI layer can multiplex sessions across slots. All per-conversation state
// (KV cache, committed ids, cursors) lives in the passed context; the session
// only keeps the scratch fill cache and the validated core_config limits.
//
// A session is not reentrant: use one session per thread.
class InferenceSession {
public:
    // Uses a default core_config derived from the model (see
    // core::default_core_config).
    explicit InferenceSession(InferenceEngine& engine);

    // Uses an already-validated core_config (see core::parse_core_config).
    InferenceSession(InferenceEngine& engine, const core::CoreConfig& core_config);

    // Zeroes the context's KV cache and clears its committed ids/cursors.
    InferenceError reset(context::Context& context);

    // Appends newly committed context-vocabulary ids using the strictly
    // causal pass. A leading BOS is accepted and ignored. History is windowed
    // to core_config.max_history_length, evicting the oldest ids down to
    // max_history_length - slack_interval (BOS retained as the attention
    // sink) when the cap would be exceeded.
    InferenceError fill(context::Context& context, const std::vector<int32_t>& new_ids);

    // Replaces the full context snapshot. Common prefixes are retained and
    // only the missing suffix is filled; divergent input is rebuilt from BOS.
    InferenceError replace_context(context::Context& context,
                                   const std::vector<int32_t>& context_ids);

    // Runs autoregressive beam search over pinyin-vocabulary ids. context_ids
    // may be supplied as the full current context; an empty vector uses the
    // context already committed in `context`. History longer than
    // max_history_length is truncated to max_history_length - slack_interval
    // first, and the pinyin window is capped at max_pinyin_length. On
    // cancellation the generation cursors are rolled back so the context is
    // left at its committed history state (the KV cache is not rewound).
    GenerateResult generate(context::Context& context,
                            const std::vector<int32_t>& pinyin_ids,
                            const std::vector<int32_t>& context_ids = {},
                            CancellationFn cancellation = nullptr,
                            void* cancellation_user_data = nullptr);

    int32_t beam_size() const { return beam_size_; }
    const core::CoreConfig& core_config() const { return core_config_; }

private:
    std::vector<int32_t> without_bos(const std::vector<int32_t>& ids) const;
    bool is_history_prefix(const std::vector<int32_t>& ids) const;
    InferenceError fill_incremental(const std::vector<int32_t>& new_ids);
    InferenceError fill_from_scratch(const std::vector<int32_t>& context_ids);
    bool cancelled(CancellationFn cancellation, void* user_data) const;
    InferenceError fill_impl(const std::vector<int32_t>& new_ids);
    InferenceError replace_context_impl(const std::vector<int32_t>& context_ids);
    int32_t history_capacity() const;
    int32_t eviction_target() const;
    // Left-shifts the committed history down to `target` committed ids,
    // keeping the first BOS as the attention sink. Updates the KV caches and
    // the cursors accordingly.
    InferenceError truncate_history(int32_t target);
    GenerateResult generate_impl(const std::vector<int32_t>& pinyin_ids,
                                 const std::vector<int32_t>& context_ids,
                                 CancellationFn cancellation, void* cancellation_user_data);
    void sync_from_context(context::Context& context);
    void sync_to_context(context::Context& context);

    InferenceEngine& engine_;
    core::CoreConfig core_config_;
    context::PersistentTensor fill_kv_;
    context::PersistentTensor self_kv_;
    std::vector<float_t> reorder_scratch_;
    std::vector<int32_t> history_ids_;
    int32_t beam_size_ = 1;
    int32_t current_seqlen_ = 0;
    int32_t history_seqlen_ = 0;
};

}  // namespace phono::engine
