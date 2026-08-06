#include "algo/trie.hpp"
#include "algo/viterbi.hpp"
#include "engine/inference_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <executorch/extension/tensor/tensor.h>

namespace phono::engine {

using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::runtime::EValue;

namespace {

std::vector<std::vector<float>> tensor_to_2d_float(const executorch::aten::Tensor& t) {
    const auto sizes = t.sizes();
    int32_t S = 0, D = 0;
    if (sizes.size() == 3) {
        S = sizes[1];
        D = sizes[2];
    } else if (sizes.size() == 2) {
        S = sizes[0];
        D = sizes[1];
    } else {
        throw std::runtime_error("tensor_to_2d_float: unexpected logits tensor rank " +
                                  std::to_string(sizes.size()));
    }

    const float* data = t.const_data_ptr<float>();
    std::vector<std::vector<float>> out(static_cast<size_t>(S), std::vector<float>(static_cast<size_t>(D)));
    for (int32_t s = 0; s < S; ++s) {
        for (int32_t d = 0; d < D; ++d) {
            out[static_cast<size_t>(s)][static_cast<size_t>(d)] = data[s * D + d];
        }
    }
    return out;
}

}  // namespace

InferenceEngine::InferenceEngine(const std::string& package_root)
    : config_(core::ModelPackageConfig::load(package_root)),
      tokenizer_(core::Tokenizer::from_config(config_)) {
    pre_module_ = std::make_unique<Module>(config_.resolve(config_.runtime.pre_model_path));
    post_module_ = std::make_unique<Module>(config_.resolve(config_.runtime.post_model_path));

    std::string pre_path = config_.resolve(config_.runtime.pre_model_path);
    pre_module_ = std::make_unique<Module>(pre_path);

    auto err = pre_module_->load_forward();
    if (err != executorch::runtime::Error::Ok) {
        throw std::runtime_error(
            "InferenceEngine: failed to load pre_model at " + pre_path +
            ", Error Code: 0x" + std::to_string(static_cast<uint32_t>(err)));
    }
    std::string post_path = config_.resolve(config_.runtime.post_model_path);
    post_module_ = std::make_unique<Module>(post_path);

    err = post_module_->load_forward();
    if (err != executorch::runtime::Error::Ok) {
        throw std::runtime_error(
            "InferenceEngine: failed to load post_model at " + post_path +
            ", Error Code: 0x" + std::to_string(static_cast<uint32_t>(err)));
    }
    
    load_trie_from_json(config_.resolve(config_.decoding.trie_path));
}

InferenceEngine::~InferenceEngine() = default;

void InferenceEngine::run_pre_model(const std::vector<int32_t>& new_prefix_ids, context::Context& ctx) {
    const int32_t new_len = static_cast<int32_t>(new_prefix_ids.size());
    const int32_t cache_pos = ctx.current_position();

    std::vector<int64_t> input_ids_data(new_prefix_ids.begin(), new_prefix_ids.end());
    int64_t pos_data = cache_pos;

    auto input_ids = make_tensor_ptr(std::vector<int32_t>{1, new_len}, input_ids_data.data(), ScalarType::Long);
    auto cache_pos_t = make_tensor_ptr(std::vector<int32_t>{1}, &pos_data, ScalarType::Long);

    std::vector<EValue> inputs = {
        EValue(*input_ids),                  // Input 0: input_ids
        EValue(),                            // Input 1: None (input_offsets)
        EValue(*ctx.pre_self_kv.tensor),     // Input 2: pre_self_kv
        EValue(*cache_pos_t),                // Input 3: current_seqlen
        EValue(),                            // Input 4: None (min_seqlen)
        EValue(),                            // Input 5: None (max_seqlen)
        EValue(*ctx.cross_kv.tensor),        // Input 6: pre_cross_kv_cache
        EValue(*cache_pos_t),                // Input 7: pre_cross_cache_pos
    };
    auto result = pre_module_->forward(inputs);
    if (!result.ok()) {
        throw std::runtime_error("InferenceEngine::run_pre_model: pre_model forward() failed (error " +
                                  std::to_string(static_cast<int>(result.error())) + ")");
    }
}

std::vector<std::vector<float>> InferenceEngine::run_post_model(const std::vector<int32_t>& postfix_ids,
                                                                context::Context& ctx) {
    const int32_t post_len = static_cast<int32_t>(postfix_ids.size());
    const int32_t total_pre = ctx.current_position();

    std::vector<int64_t> postfix_ids_data(postfix_ids.begin(), postfix_ids.end());
    int64_t total_pre_data = total_pre;

    auto input_ids = make_tensor_ptr(std::vector<int32_t>{1, post_len}, postfix_ids_data.data(), ScalarType::Long);
    auto total_pre_t = make_tensor_ptr(std::vector<int32_t>{1}, &total_pre_data, ScalarType::Long);

    std::vector<EValue> inputs = {
        EValue(*input_ids),                // 0: input_ids
        EValue(),                          // 1: input_offsets (None)
        EValue(),                          // 2: pre_K (None)
        EValue(),                          // 3: pre_V (None)
        EValue(),                          // 4: pre_offsets (None)
        EValue(*ctx.cross_kv.tensor),      // 5: pre_cross_kv_cache
        EValue(*total_pre_t),              // 6: current_seqlen
        EValue(),                          // 7: min_seqlen (None)
        EValue(),                          // 8: max_seqlen (None)
        EValue(),                          // 9: min_seqlen_pre (None)
        EValue(),                          // 10: max_seqlen_pre (None)
        EValue(false),                     // 11: return_last_hidden (False)
    };

    auto result = post_module_->forward(inputs);
    if (!result.ok()) {
        throw std::runtime_error("InferenceEngine::run_post_model: post_model forward() failed (error " +
                                  std::to_string(static_cast<int>(result.error())) + ")");
    }

    const auto& logits_tensor = result->at(0).toTensor();
    return tensor_to_2d_float(logits_tensor);
}

InferenceSession::InferenceSession(InferenceEngine& engine) : engine_(engine) {}

void InferenceSession::initialize_context(context::Context& ctx) {
    // The KV caches of a fresh context are zeroed; write BOS at position 0.
    // BOS is intentionally NOT recorded in ctx.context_ids()
    ctx.set_current_position(0);
    const auto& tok = engine_.tokenizer();
    const int32_t bos_id = tok.special_token_id("bos_token");
    std::vector<int32_t> bos{ bos_id };
    engine_.run_pre_model(bos, ctx);
    ctx.set_current_position(1);
}

void InferenceSession::advance_context(context::Context& ctx, const std::string& new_text) {
    if (ctx.current_position() == 0) {
        // Fresh/cleared context: the KV caches are zeroed, so re-seed BOS
        // before continuing to stream.
        initialize_context(ctx);
    }

    const auto& tok = engine_.tokenizer();
    std::vector<int32_t> new_ids = tok.encode_context(new_text);
    if (new_ids.empty()) {
        return;  // nothing new to feed
    }

    ctx.append_context_ids(new_ids.data(), static_cast<int32_t>(new_ids.size()));
    engine_.run_pre_model(new_ids, ctx);
    ctx.set_current_position(ctx.current_position() + static_cast<int32_t>(new_ids.size()));
}

TopKStepResult InferenceSession::predict_step(context::Context& ctx, const std::string& text_utf8,
                                               const std::vector<std::string>& pinyin_list) {
    advance_context(ctx, text_utf8);
    const auto& tok = engine_.tokenizer();

    auto postfix_ids = tok.encode_pinyin(pinyin_list);
    auto logits = engine_.run_post_model(postfix_ids, ctx);

    TopKStepResult result;
    result.current_seqlen = ctx.current_position();
    result.pred_ids.reserve(logits.size());
    for (const auto& row : logits) {
        int32_t best_id = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i] > best_val) {
                best_val = row[i];
                best_id = static_cast<int32_t>(i);
            }
        }
        result.pred_ids.push_back(best_id);
    }
    result.decoded = tok.ids_to_text(result.pred_ids);
    return result;
}

TopNStepResult InferenceSession::predict_step_topn(context::Context& ctx, const std::string& text_utf8,
                                                   const std::vector<std::string>& pinyin_list, int topk) {
    advance_context(ctx, text_utf8);
    const auto& tok = engine_.tokenizer();

    auto postfix_ids = tok.encode_pinyin(pinyin_list);
    auto logits = engine_.run_post_model(postfix_ids, ctx);

    TopNStepResult result;
    result.current_seqlen = ctx.current_position();
    const size_t S = logits.size();
    result.pred_ids_per_pos.resize(S);
    result.decoded_per_pos.resize(S);
    result.probs_per_pos.resize(S);
    result.logits_per_pos.resize(S);
    result.entropy_per_pos.resize(S);

    for (size_t s = 0; s < S; ++s) {
        const auto& row = logits[s];
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float v : row) max_logit = std::max(max_logit, v);

        std::vector<float> probs(row.size());
        double sum = 0.0;
        for (size_t i = 0; i < row.size(); ++i) {
            probs[i] = std::exp(row[i] - max_logit);
            sum += probs[i];
        }
        for (auto& p : probs) p = static_cast<float>(p / sum);

        double entropy = 0.0;
        for (float p : probs) {
            if (p > 0.0f) entropy -= static_cast<double>(p) * std::log(static_cast<double>(p));
        }
        result.entropy_per_pos[s] = static_cast<float>(entropy);

        std::vector<size_t> idx(row.size());
        std::iota(idx.begin(), idx.end(), 0);
        const int k = std::min<int>(topk, static_cast<int>(idx.size()));
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                           [&](size_t a, size_t b) { return row[a] > row[b]; });

        result.pred_ids_per_pos[s].reserve(static_cast<size_t>(k));
        result.decoded_per_pos[s].reserve(static_cast<size_t>(k));
        result.probs_per_pos[s].reserve(static_cast<size_t>(k));
        result.logits_per_pos[s].reserve(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i) {
            const size_t id = idx[static_cast<size_t>(i)];
            result.pred_ids_per_pos[s].push_back(static_cast<int32_t>(id));
            result.decoded_per_pos[s].push_back(tok.id_to_chinese(static_cast<int32_t>(id)));
            result.probs_per_pos[s].push_back(probs[id]);
            result.logits_per_pos[s].push_back(row[id]);
        }
    }
    return result;
}

FullLogitsStepResult InferenceSession::predict_step_full_logits(context::Context& ctx,
                                                                 const std::string& text_utf8,
                                                                 const std::vector<std::string>& pinyin_list) {
    advance_context(ctx, text_utf8);
    const auto& tok = engine_.tokenizer();

    auto postfix_ids = tok.encode_pinyin(pinyin_list);
    FullLogitsStepResult result;
    result.logits = engine_.run_post_model(postfix_ids, ctx);
    result.current_seqlen = ctx.current_position();
    return result;
}

ViterbiStepResult InferenceSession::predict_step_viterbi(context::Context& ctx, const std::string& text_utf8,
                                                         const std::vector<std::string>& pinyin_list,
                                                         double beta_single, double beta_word, double epsilon,
                                                         int n_best) {
    FullLogitsStepResult full = predict_step_full_logits(ctx, text_utf8, pinyin_list);
    const auto& tok = engine_.tokenizer();

    const size_t S = full.logits.size();
    std::vector<std::unordered_map<std::string, double>> candidates(S);
    for (size_t s = 0; s < S; ++s) {
        const auto& row = full.logits[s];
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float v : row) max_logit = std::max(max_logit, v);

        std::vector<double> probs(row.size());
        double sum = 0.0;
        for (size_t i = 0; i < row.size(); ++i) {
            probs[i] = std::exp(static_cast<double>(row[i] - max_logit));
            sum += probs[i];
        }
        for (size_t i = 0; i < row.size(); ++i) {
            const double p = probs[i] / sum;
            if (p > epsilon) {
                const std::string ch = tok.id_to_chinese(static_cast<int32_t>(i));
                if (!ch.empty()) {
                    candidates[s][ch] = p;
                }
            }
        }
    }

    const nlohmann::json& trie = engine_.trie();
    std::vector<std::vector<algo::WordMatch>> words_at(S);
    for (size_t s = 0; s < S; ++s) {
        words_at[s] = algo::find_matching_words(trie, candidates, static_cast<int>(s));
    }

    auto best = algo::viterbi_nbest(candidates, words_at, beta_single, beta_word, n_best);

    ViterbiStepResult result;
    result.current_seqlen = full.current_seqlen;
    result.nbest.reserve(best.size());
    for (auto& b : best) {
        NBestEntry entry;
        entry.score = b.score;
        entry.words = b.words;
        for (const auto& w : b.words) entry.text += w;
        result.nbest.push_back(std::move(entry));
    }
    return result;
}

}  // namespace phono::engine
