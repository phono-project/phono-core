// inference_engine.hpp
#pragma once

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>

#include "algo/trie.hpp"
#include "context/context.hpp"
#include "context/kv_cache.hpp"
#include "core/config.hpp"
#include "core/tokenizer.hpp"

namespace phono::engine {

// predict_step results

struct TopKStepResult {
    std::vector<int32_t> pred_ids;   // per-position argmax id (topk==1)
    std::string decoded;             // ids_to_text(pred_ids)
    int32_t current_seqlen = 0;
};

struct TopNStepResult {
    // Per post-position, top-k candidate ids/probs/logits (topk>1), plus entropy.
    std::vector<std::vector<int32_t>> pred_ids_per_pos;
    std::vector<std::vector<std::string>> decoded_per_pos;
    std::vector<std::vector<float>> probs_per_pos;
    std::vector<std::vector<float>> logits_per_pos;
    std::vector<float> entropy_per_pos;
    int32_t current_seqlen = 0;
};

// Full per-position logits (topk==0), used internally by predict_step_viterbi
// but also exposed for callers who want to post-process differently.
struct FullLogitsStepResult {
    std::vector<std::vector<float>> logits;  // [S_post][proj_size]
    int32_t current_seqlen = 0;
};

struct NBestEntry {
    double score = 0.0;
    std::vector<std::string> words;  // ordered chars / dictionary words
    std::string text;                // words joined together
};

struct ViterbiStepResult {
    std::vector<NBestEntry> nbest;
    int32_t current_seqlen = 0;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& package_root);
    ~InferenceEngine();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    const core::ModelPackageConfig& config() const { return config_; }
    const core::Tokenizer& tokenizer() const { return tokenizer_; }

    const nlohmann::json& trie() const { return trie_; }
    void load_trie_from_json(const std::string& path) { trie_ = algo::load_trie(path); }

    // Runs the pre model over the *new* prefix tokens for `ctx`, writing
    // them at the ctx's current_position cursor (which advance_context has
    // already advanced past). Mutates ctx's KV-cache views in place.
    void run_pre_model(const std::vector<int32_t>& new_prefix_ids, context::Context& ctx);

    // Runs the post model over the pinyin `postfix_ids`, cross-attending to
    // the ctx's KV caches up to ctx.current_position(). Returns
    // [S_post][proj_size] logits.
    std::vector<std::vector<float>> run_post_model(const std::vector<int32_t>& postfix_ids,
                                                   context::Context& ctx);

private:
    core::ModelPackageConfig config_;
    core::Tokenizer tokenizer_;
    std::unique_ptr<executorch::extension::Module> pre_module_;
    std::unique_ptr<executorch::extension::Module> post_module_;
    nlohmann::json trie_;
};

// Stateless streaming runner. All per-conversation state lives in the
// `Context` passed in, so one session can drive many contexts.
class InferenceSession {
public:
    explicit InferenceSession(InferenceEngine& engine);

    // Seeds BOS into `ctx`'s KV caches at position 0 and sets its
    // current_position to 1. Call once per fresh Context (advance_context
    // does this automatically when the context is still empty).
    void initialize_context(context::Context& ctx);

    // Advances `ctx` by appending the NEW context text `new_text` — i.e. the
    // characters committed since the last call, not the full accumulated
    // context. Only those new tokens are encoded and run through the pre
    // model; the KV caches carry everything already fed. The new tokenizer
    // ids are also recorded in the context. Treating the input as strictly
    // new text keeps this composable with higher-level management (e.g.
    // truncating the head of the context to slide the window).
    void advance_context(context::Context& ctx, const std::string& new_text);

    // greedy per-position Chinese-character prediction from `pinyin_list`, given the text
    // committed since the last call (`new_text`), which is appended to the Context before decoding.
    TopKStepResult predict_step(context::Context& ctx, const std::string& new_text,
                                const std::vector<std::string>& pinyin_list);

    // top-k candidates per position plus entropy.
    TopNStepResult predict_step_topn(context::Context& ctx, const std::string& new_text,
                                     const std::vector<std::string>& pinyin_list, int topk);

    // full per-position logits.
    FullLogitsStepResult predict_step_full_logits(context::Context& ctx, const std::string& new_text,
                                                  const std::vector<std::string>& pinyin_list);

    // full logits -> softmax -> epsilon-filtered per-position candidates ->
    // trie word matching -> Viterbi N-best dictionary-constrained decoding.
    ViterbiStepResult predict_step_viterbi(context::Context& ctx, const std::string& new_text,
                                           const std::vector<std::string>& pinyin_list,
                                           double beta_single, double beta_word, double epsilon,
                                           int n_best);

private:
    InferenceEngine& engine_;
};

}  // namespace phono::engine

