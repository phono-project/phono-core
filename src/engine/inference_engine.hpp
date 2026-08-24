#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

#include "algo/trie.hpp"
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
    std::vector<uint8_t> logits_mask;
    int32_t batch_size = 0;
    int32_t sequence_length = 0;
    int32_t hidden_dim = 0;
    int32_t projection_size = 0;
};

struct DecoderModelOutput {
    std::vector<float> logits;
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

    void load_trie_from_json(const std::string& path) { trie_ = algo::load_trie(path); }
    const nlohmann::json& trie() const { return trie_; }

    // These methods execute the v2 exported methods. They return the runtime
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
    nlohmann::json trie_;
    int32_t pre_pass1_batch_size_ = 0;
    int32_t pre_pass2_batch_size_ = 0;
};

class InferenceSession {
public:
    explicit InferenceSession(InferenceEngine& engine);
    InferenceSession(InferenceEngine& engine, int32_t beam_size);
    InferenceSession(InferenceEngine& engine, context::Context& context);

    // Allocates no per-call cache. The model-configured B dimension is used as
    // the beam/batch width for the session.
    InferenceError initialize();
    InferenceError reset();

    // Appends newly committed context-vocabulary ids using the strictly
    // causal pass. A leading BOS is accepted and ignored.
    InferenceError fill(const std::vector<int32_t>& new_ids);

    // Replaces the full context snapshot. Common prefixes are retained and
    // only the missing suffix is filled; divergent input is rebuilt from BOS.
    InferenceError replace_context(const std::vector<int32_t>& context_ids);

    // Runs autoregressive beam search over pinyin-vocabulary ids. context_ids
    // may be supplied as the full current context; an empty vector uses the
    // context already filled into this session. The history cache is never
    // advanced by generation.
    GenerateResult generate(const std::vector<int32_t>& pinyin_ids,
                            const std::vector<int32_t>& context_ids = {},
                            const std::atomic_bool* cancellation = nullptr);
    GenerateResult generate(const std::vector<int32_t>& pinyin_ids,
                            const std::vector<int32_t>& context_ids,
                            const std::atomic_bool& cancellation) {
        return generate(pinyin_ids, context_ids, &cancellation);
    }

    int32_t current_seqlen() const { return current_seqlen_; }
    int32_t history_seqlen() const { return history_seqlen_; }
    int32_t beam_size() const { return beam_size_; }

private:
    std::vector<int32_t> without_bos(const std::vector<int32_t>& ids) const;
    bool is_history_prefix(const std::vector<int32_t>& ids) const;
    InferenceError fill_incremental(const std::vector<int32_t>& new_ids);
    InferenceError fill_from_scratch(const std::vector<int32_t>& context_ids);
    bool cancelled(const std::atomic_bool* cancellation) const;
    InferenceError fill_impl(const std::vector<int32_t>& new_ids);
    InferenceError replace_context_impl(const std::vector<int32_t>& context_ids);
    GenerateResult generate_impl(const std::vector<int32_t>& pinyin_ids,
                                 const std::vector<int32_t>& context_ids,
                                 const std::atomic_bool* cancellation);
    void sync_from_context();
    void sync_to_context();

    InferenceEngine& engine_;
    context::Context* context_ = nullptr;
    context::PersistentTensor self_kv_;
    context::PersistentTensor fill_kv_;
    std::vector<int32_t> history_ids_;
    int32_t beam_size_ = 1;
    int32_t current_seqlen_ = 0;
    int32_t history_seqlen_ = 0;
};

}  // namespace phono::engine
