#include "engine/inference_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <executorch/extension/tensor/tensor.h>

namespace phono::engine {

using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::runtime::EValue;

namespace {

std::string runtime_error_message(executorch::runtime::Error error) {
    return std::to_string(static_cast<int>(error));
}

bool all_equal(const std::vector<int32_t>& values) {
    return values.empty() || std::all_of(values.begin() + 1, values.end(),
                                          [&](int32_t value) { return value == values.front(); });
}

int32_t method_batch_size(const executorch::runtime::MethodMeta& meta, size_t input_index) {
    auto tensor_meta = meta.input_tensor_meta(input_index);
    if (!tensor_meta.ok() || tensor_meta->sizes().empty()) {
        return 0;
    }
    const int32_t batch = tensor_meta->sizes()[0];
    return batch > 0 ? batch : 0;
}

void validate_method(Module& module, const std::string& name,
                     const std::vector<executorch::runtime::Tag>& input_tags,
                     const std::vector<executorch::runtime::Tag>& output_tags) {
    auto meta_result = module.method_meta(name);
    if (!meta_result.ok()) {
        throw std::runtime_error("InferenceEngine: cannot inspect method " + name);
    }
    const auto& meta = *meta_result;
    if (meta.num_inputs() != input_tags.size() || meta.num_outputs() != output_tags.size()) {
        throw std::runtime_error("InferenceEngine: unexpected input/output count for method " + name);
    }
    for (size_t i = 0; i < input_tags.size(); ++i) {
        auto tag = meta.input_tag(i);
        if (!tag.ok() || tag.get() != input_tags[i]) {
            throw std::runtime_error("InferenceEngine: unexpected input type for method " + name);
        }
    }
    for (size_t i = 0; i < output_tags.size(); ++i) {
        auto tag = meta.output_tag(i);
        if (!tag.ok() || tag.get() != output_tags[i]) {
            throw std::runtime_error("InferenceEngine: unexpected output type for method " + name);
        }
    }
}

struct TokenScore {
    int32_t id = 0;
    double log_prob = -std::numeric_limits<double>::infinity();
};

std::vector<TokenScore> top_tokens(const std::vector<float>& logits, size_t row_start,
                                   int32_t projection_size, int32_t count) {
    if (count <= 0 || projection_size <= 0) {
        return {};
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < projection_size; ++i) {
        max_logit = std::max(max_logit, logits[row_start + static_cast<size_t>(i)]);
    }
    if (!std::isfinite(max_logit)) {
        return {};
    }

    double exp_sum = 0.0;
    for (int32_t i = 0; i < projection_size; ++i) {
        exp_sum += std::exp(static_cast<double>(logits[row_start + static_cast<size_t>(i)]) -
                            static_cast<double>(max_logit));
    }
    if (!(exp_sum > 0.0)) {
        return {};
    }
    const double log_norm = static_cast<double>(max_logit) + std::log(exp_sum);

    std::vector<TokenScore> candidates;
    candidates.reserve(static_cast<size_t>(projection_size));
    for (int32_t i = 0; i < projection_size; ++i) {
        const float logit = logits[row_start + static_cast<size_t>(i)];
        if (std::isfinite(logit)) {
            candidates.push_back(TokenScore{i, static_cast<double>(logit) - log_norm});
        }
    }

    const int32_t keep = std::min<int32_t>(count, static_cast<int32_t>(candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                      [](const TokenScore& lhs, const TokenScore& rhs) {
                          if (lhs.log_prob != rhs.log_prob) return lhs.log_prob > rhs.log_prob;
                          return lhs.id < rhs.id;
                      });
    candidates.resize(static_cast<size_t>(keep));
    return candidates;
}

}  // namespace

const char* inference_error_name(InferenceError error) {
    switch (error) {
        case InferenceError::Ok: return "ok";
        case InferenceError::InvalidArgument: return "invalid_argument";
        case InferenceError::ContextLimitExceeded: return "context_limit_exceeded";
        case InferenceError::PinyinLimitExceeded: return "pinyin_limit_exceeded";
        case InferenceError::NoCandidates: return "no_candidates";
        case InferenceError::Cancelled: return "cancelled";
        case InferenceError::ModelError: return "model_error";
    }
    return "unknown";
}

InferenceEngine::InferenceEngine(const std::string& package_root)
    : config_(core::ModelPackageConfig::load(package_root)),
      tokenizer_(core::Tokenizer::from_config(config_)),
      pre_module_(std::make_unique<Module>(config_.resolve(config_.runtime.pre_model_path))),
      post_module_(std::make_unique<Module>(config_.resolve(config_.runtime.post_model_path))) {
    auto pre_methods = pre_module_->method_names();
    if (!pre_methods.ok() ||
        pre_methods->count(config_.runtime.pre_pass1_method) == 0 ||
        pre_methods->count(config_.runtime.pre_pass2_method) == 0) {
        throw std::runtime_error(
            "InferenceEngine: pre model does not contain configured v2.1 methods");
    }

    auto post_methods = post_module_->method_names();
    if (!post_methods.ok() || post_methods->count(config_.runtime.post_method) == 0) {
        throw std::runtime_error(
            "InferenceEngine: post model does not contain configured method");
    }

    using executorch::runtime::Tag;
    auto pre1_meta = pre_module_->method_meta(config_.runtime.pre_pass1_method);
    auto pre2_meta = pre_module_->method_meta(config_.runtime.pre_pass2_method);
    pre_pass1_cache_only_ = pre1_meta->num_inputs() == 5 && pre1_meta->num_outputs() == 1;
    if (pre_pass1_cache_only_) {
        validate_method(*pre_module_, config_.runtime.pre_pass1_method,
                        {Tag::Tensor, Tag::Tensor, Tag::Tensor, Tag::Bool, Tag::Bool},
                        {Tag::Tensor});
    } else {
        validate_method(*pre_module_, config_.runtime.pre_pass1_method,
                        {Tag::Tensor, Tag::Tensor, Tag::Tensor, Tag::Bool},
                        {Tag::Tensor, Tag::Tensor});
    }
    validate_method(*pre_module_, config_.runtime.pre_pass2_method,
                    {Tag::Tensor, Tag::Tensor, Tag::Tensor, Tag::Tensor,
                     Tag::Tensor, Tag::Tensor, Tag::Tensor, Tag::Bool},
                    {Tag::Tensor, Tag::Tensor});
    validate_method(*post_module_, config_.runtime.post_method,
                    {Tag::Tensor}, {Tag::Tensor, Tag::Tensor, Tag::Tensor});
    pre_pass1_batch_size_ = method_batch_size(*pre1_meta, 0);
    pre_pass2_batch_size_ = method_batch_size(*pre2_meta, 0);

    auto err = pre_module_->load_method(config_.runtime.pre_pass1_method);
    if (err != executorch::runtime::Error::Ok) {
        throw std::runtime_error("InferenceEngine: failed to load pre pass 1 (error " +
                                 runtime_error_message(err) + ")");
    }
    err = pre_module_->load_method(config_.runtime.pre_pass2_method);
    if (err != executorch::runtime::Error::Ok) {
        throw std::runtime_error("InferenceEngine: failed to load pre pass 2 (error " +
                                 runtime_error_message(err) + ")");
    }
    err = post_module_->load_method(config_.runtime.post_method);
    if (err != executorch::runtime::Error::Ok) {
        throw std::runtime_error("InferenceEngine: failed to load post model (error " +
                                 runtime_error_message(err) + ")");
    }

}

InferenceEngine::~InferenceEngine() = default;

InferenceError InferenceEngine::runtime_error_to_status(executorch::runtime::Error error) {
    return error == executorch::runtime::Error::Ok ? InferenceError::Ok
                                                   : InferenceError::ModelError;
}

InferenceError InferenceEngine::run_pre_pass1(const std::vector<int32_t>& input_ids,
                                              context::PersistentTensor& self_kv,
                                              int32_t current_seqlen,
                                              int32_t batch_size) const {
    if (batch_size <= 0 || input_ids.empty() || input_ids.size() % static_cast<size_t>(batch_size) != 0) {
        return InferenceError::InvalidArgument;
    }
    const int32_t sequence_length = static_cast<int32_t>(input_ids.size() / batch_size);
    std::vector<int64_t> input_data(input_ids.begin(), input_ids.end());
    int64_t position_data = current_seqlen;

    auto input = make_tensor_ptr(std::vector<int32_t>{batch_size, sequence_length},
                                 input_data.data(), ScalarType::Long);
    auto position = make_tensor_ptr(std::vector<int32_t>{1}, &position_data, ScalarType::Long);
    std::vector<EValue> inputs = {
        EValue(*input),
        EValue(*self_kv.tensor),
        EValue(*position),
        EValue(true),
    };
    if (pre_pass1_cache_only_) {
        inputs.emplace_back(false);
    }
    auto result = pre_module_->execute(config_.runtime.pre_pass1_method, inputs);
    return runtime_error_to_status(result.error());
}

InferenceError InferenceEngine::run_pre_pass2(const std::vector<int32_t>& input_ids,
                                              context::PersistentTensor& self_kv,
                                              const std::vector<int32_t>& current_seqlen,
                                              const PostModelOutput& post,
                                              int32_t cross_q_pos_start,
                                              DecoderModelOutput& output) const {
    const int32_t batch_size = static_cast<int32_t>(input_ids.size());
    if (batch_size <= 0 || !all_equal(current_seqlen) ||
        post.sequence_length <= 0 || post.hidden_dim <= 0 || post.candidate_width <= 0 ||
        post.batch_size != 1 ||
        post.hidden.size() != static_cast<size_t>(post.sequence_length) * post.hidden_dim ||
        post.candidate_ids.size() !=
            static_cast<size_t>(post.sequence_length) * post.candidate_width ||
        post.candidate_mask.size() != post.candidate_ids.size() ||
        cross_q_pos_start < 0 || cross_q_pos_start >= post.sequence_length) {
        return InferenceError::InvalidArgument;
    }
    if (self_kv.batch_size() != batch_size) {
        return InferenceError::InvalidArgument;
    }

    std::vector<int64_t> input_data(input_ids.begin(), input_ids.end());
    std::vector<int64_t> position_data{current_seqlen.front()};
    std::vector<float> hidden_data(
        static_cast<size_t>(batch_size) * post.hidden.size());
    std::vector<int64_t> candidate_ids;
    std::vector<int32_t> mapped_candidate_ids;
    for (int32_t batch = 0; batch < batch_size; ++batch) {
        std::copy(post.hidden.begin(), post.hidden.end(),
                  hidden_data.begin() + static_cast<size_t>(batch) * post.hidden.size());
    }
    const size_t candidate_start =
        static_cast<size_t>(cross_q_pos_start) * post.candidate_width;
    candidate_ids.reserve(static_cast<size_t>(post.candidate_width));
    mapped_candidate_ids.reserve(static_cast<size_t>(post.candidate_width));
    for (int32_t index = 0; index < post.candidate_width; ++index) {
        if (post.candidate_mask[candidate_start + static_cast<size_t>(index)] == 0) {
            continue;
        }
        const int32_t id = post.candidate_ids[candidate_start + static_cast<size_t>(index)];
        candidate_ids.push_back(id);
        mapped_candidate_ids.push_back(id);
    }
    if (candidate_ids.empty()) {
        return InferenceError::NoCandidates;
    }
    int64_t post_position_offset = 1;
    int64_t query_position = cross_q_pos_start;

    auto input = make_tensor_ptr(std::vector<int32_t>{batch_size, 1}, input_data.data(),
                                 ScalarType::Long);
    auto position = make_tensor_ptr(std::vector<int32_t>{1}, position_data.data(),
                                    ScalarType::Long);
    auto hidden = make_tensor_ptr(
        std::vector<int32_t>{batch_size, post.sequence_length, post.hidden_dim},
        hidden_data.data(), ScalarType::Float);
    auto post_offset = make_tensor_ptr(std::vector<int32_t>{}, &post_position_offset,
                                       ScalarType::Long);
    auto query_offset = make_tensor_ptr(std::vector<int32_t>{}, &query_position,
                                        ScalarType::Long);
    auto candidates = make_tensor_ptr(
        std::vector<int32_t>{static_cast<int32_t>(candidate_ids.size())},
        candidate_ids.data(), ScalarType::Long);

    std::vector<EValue> inputs = {
        EValue(*input),
        EValue(*self_kv.tensor),
        EValue(*position),
        EValue(*hidden),
        EValue(*post_offset),
        EValue(*query_offset),
        EValue(*candidates),
        EValue(true),
    };
    auto result = pre_module_->execute(config_.runtime.pre_pass2_method, inputs);
    if (!result.ok() || result->empty()) {
        return result.ok() ? InferenceError::ModelError
                           : runtime_error_to_status(result.error());
    }

    const auto& logits = result->at(0).toTensor();
    const auto sizes = logits.sizes();
    if (sizes.size() != 3 || sizes[0] != batch_size || sizes[1] != 1) {
        return InferenceError::ModelError;
    }
    output.batch_size = batch_size;
    output.sequence_length = 1;
    output.projection_size = static_cast<int32_t>(sizes[2]);
    if (output.projection_size != static_cast<int32_t>(candidate_ids.size())) {
        return InferenceError::ModelError;
    }
    output.candidate_ids = std::move(mapped_candidate_ids);
    const float* data = logits.const_data_ptr<float>();
    output.logits.assign(data, data + static_cast<size_t>(batch_size) * sizes[2]);
    return InferenceError::Ok;
}

InferenceError InferenceEngine::run_post_model(const std::vector<int32_t>& pinyin_ids,
                                               PostModelOutput& output) const {
    if (pinyin_ids.empty()) {
        return InferenceError::InvalidArgument;
    }
    // The exported post program has a hard input length baked into the graph;
    // feeding anything longer is a model error on the caller's part.
    if (static_cast<int32_t>(pinyin_ids.size()) > config_.post_model.max_seqlen) {
        return InferenceError::PinyinLimitExceeded;
    }
    std::vector<int64_t> input_data(pinyin_ids.begin(), pinyin_ids.end());
    auto input = make_tensor_ptr(std::vector<int32_t>{1, static_cast<int32_t>(pinyin_ids.size())},
                                 input_data.data(), ScalarType::Long);
    auto result = post_module_->execute(config_.runtime.post_method, {EValue(*input)});
    if (!result.ok() || result->size() < 3) {
        return result.ok() ? InferenceError::ModelError
                           : runtime_error_to_status(result.error());
    }

    const auto& hidden = result->at(0).toTensor();
    const auto& candidates = result->at(1).toTensor();
    const auto& mask = result->at(2).toTensor();
    const auto hidden_sizes = hidden.sizes();
    const auto candidate_sizes = candidates.sizes();
    const auto mask_sizes = mask.sizes();
    if (hidden_sizes.size() != 3 || candidate_sizes.size() != 3 || mask_sizes.size() != 3 ||
        hidden_sizes[0] != 1 || candidate_sizes[0] != 1 || mask_sizes[0] != 1 ||
        hidden_sizes[1] != candidate_sizes[1] || candidate_sizes != mask_sizes) {
        return InferenceError::ModelError;
    }

    output.batch_size = 1;
    output.sequence_length = static_cast<int32_t>(hidden_sizes[1]);
    output.hidden_dim = static_cast<int32_t>(hidden_sizes[2]);
    output.candidate_width = static_cast<int32_t>(candidate_sizes[2]);
    const float* hidden_data = hidden.const_data_ptr<float>();
    output.hidden.assign(hidden_data,
                         hidden_data + static_cast<size_t>(output.sequence_length) * output.hidden_dim);
    const int64_t* candidate_data = candidates.const_data_ptr<int64_t>();
    const bool* mask_data = mask.const_data_ptr<bool>();
    const size_t candidate_count =
        static_cast<size_t>(output.sequence_length) * output.candidate_width;
    output.candidate_ids.assign(candidate_data, candidate_data + candidate_count);
    output.candidate_mask.resize(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
        output.candidate_mask[i] = mask_data[i] ? 1 : 0;
    }
    return InferenceError::Ok;
}

InferenceSession::InferenceSession(InferenceEngine& engine)
    : InferenceSession(engine, core::default_core_config(engine.config())) {}

InferenceSession::InferenceSession(InferenceEngine& engine, const core::CoreConfig& core_config)
    : engine_(engine), core_config_(core_config), beam_size_(core_config.beam_size) {
    if (core_config_.beam_size <= 0) {
        throw std::invalid_argument("InferenceSession: core_config.beam_size must be positive");
    }
    if (engine_.pre_pass2_batch_size() > 0 &&
        engine_.pre_pass2_batch_size() != core_config_.beam_size) {
        throw std::invalid_argument(
            "InferenceSession: core_config.beam_size must match pre pass 2");
    }
    const auto& cfg = engine_.config();
    fill_kv_ = context::make_zero_persistent_tensor(
        {cfg.pre_model.mhsa_layers, 2, 1, cfg.pre_model.max_seqlen,
         cfg.pre_model.mhsa_heads, cfg.pre_model.self_head_dim()});
}

InferenceError InferenceSession::reset(context::Context& context) {
    context.truncate_context_ids(0);
    context.set_current_seqlen(0);
    context.set_history_seqlen(0);
    return InferenceError::Ok;
}

std::vector<int32_t> InferenceSession::without_bos(const std::vector<int32_t>& ids) const {
    if (ids.empty()) {
        return {};
    }
    const int32_t bos = engine_.tokenizer().special_token_id("bos_token");
    if (ids.front() == bos) {
        return std::vector<int32_t>(ids.begin() + 1, ids.end());
    }
    return ids;
}

bool InferenceSession::is_history_prefix(const std::vector<int32_t>& ids) const {
    return ids.size() >= history_ids_.size() &&
           std::equal(history_ids_.begin(), history_ids_.end(), ids.begin());
}

int32_t InferenceSession::history_capacity() const {
    return core_config_.max_history_length;
}

int32_t InferenceSession::eviction_target() const {
    return core_config_.max_history_length - core_config_.slack_interval;
}

InferenceError InferenceSession::truncate_history(int32_t target) {
    const int32_t current = static_cast<int32_t>(history_ids_.size());
    if (target < 0 || target >= current) {
        return InferenceError::InvalidArgument;
    }
    const int32_t discarded = current - target;
    for (int32_t batch = 0; batch < beam_size_; ++batch) {
        self_kv_.shift_batch_tokens(batch, discarded, target);
    }
    fill_kv_.shift_batch_tokens(0, discarded, target);
    history_ids_.erase(history_ids_.begin(), history_ids_.begin() + discarded);
    history_seqlen_ = 1 + target;
    current_seqlen_ = history_seqlen_;
    return InferenceError::Ok;
}

InferenceError InferenceSession::fill_incremental(const std::vector<int32_t>& new_ids) {
    if (new_ids.empty()) {
        return InferenceError::Ok;
    }
    const int32_t capacity = history_capacity();
    const int32_t target = eviction_target();
    const int32_t old_content = static_cast<int32_t>(history_ids_.size());
    const int32_t total = old_content + static_cast<int32_t>(new_ids.size());
    if (total > capacity) {
        // The committed history would exceed max_history_length. Evict the
        // oldest ids down to max_history_length - slack_interval (keeping the
        // first BOS as the attention sink), then append the new ids.
        const int32_t keep = target - static_cast<int32_t>(new_ids.size());
        if (keep <= 0) {
            const std::vector<int32_t> tail(new_ids.end() - target, new_ids.end());
            return fill_from_scratch(tail);
        }
        const int32_t discarded = old_content - keep;
        for (int32_t batch = 0; batch < beam_size_; ++batch) {
            self_kv_.shift_batch_tokens(batch, discarded, keep);
        }
        fill_kv_.shift_batch_tokens(0, discarded, keep);
        history_ids_.erase(history_ids_.begin(), history_ids_.begin() + discarded);
        history_seqlen_ = 1 + keep;
        current_seqlen_ = history_seqlen_;
    }

    const int32_t prefill_batch = engine_.pre_pass1_batch_size();
    if (prefill_batch != 0 && prefill_batch != 1 && prefill_batch != beam_size_) {
        return InferenceError::InvalidArgument;
    }
    const int32_t input_batch = prefill_batch == 0 ? 1 : prefill_batch;
    std::vector<int32_t> input_ids(static_cast<size_t>(input_batch) * new_ids.size());
    for (int32_t batch = 0; batch < input_batch; ++batch) {
        std::copy(new_ids.begin(), new_ids.end(),
                  input_ids.begin() + static_cast<size_t>(batch) * new_ids.size());
    }
    context::PersistentTensor& prefill_cache = input_batch == 1 ? fill_kv_ : self_kv_;
    if (input_batch == 1 && history_seqlen_ > 0) {
        // A session may move between Context slots. Synchronize the clean
        // causal prefix only when pass 1 will actually consume it; generation
        // itself reads self_kv_ and should not pay for this copy.
        fill_kv_.copy_batch_slice_from(self_kv_, 0, 0, 0, history_seqlen_);
    }
    const InferenceError error = engine_.run_pre_pass1(
        input_ids, prefill_cache, history_seqlen_, input_batch);
    if (error != InferenceError::Ok) {
        return error;
    }

    if (input_batch == 1) {
        self_kv_.copy_batch_slice_from(
            fill_kv_, 0, 0, history_seqlen_, static_cast<int32_t>(new_ids.size()));
        self_kv_.copy_batch_slice_to_all(0, history_seqlen_,
                                         static_cast<int32_t>(new_ids.size()));
    }
    history_ids_.insert(history_ids_.end(), new_ids.begin(), new_ids.end());
    history_seqlen_ += static_cast<int32_t>(new_ids.size());
    current_seqlen_ = history_seqlen_;
    return InferenceError::Ok;
}

InferenceError InferenceSession::fill_from_scratch(const std::vector<int32_t>& context_ids) {
    const int32_t capacity = history_capacity();
    const int32_t target = eviction_target();
    std::vector<int32_t> effective_ids = context_ids;
    if (effective_ids.size() > static_cast<size_t>(capacity)) {
        effective_ids.erase(effective_ids.begin(), effective_ids.end() - target);
    }
    history_ids_.clear();
    current_seqlen_ = 0;
    history_seqlen_ = 0;
    if (effective_ids.empty()) {
        return InferenceError::Ok;
    }

    const int32_t bos = engine_.tokenizer().special_token_id("bos_token");
    std::vector<int32_t> causal_ids;
    causal_ids.reserve(effective_ids.size() + 1);
    causal_ids.push_back(bos);
    causal_ids.insert(causal_ids.end(), effective_ids.begin(), effective_ids.end());
    const int32_t prefill_batch = engine_.pre_pass1_batch_size();
    if (prefill_batch != 0 && prefill_batch != 1 && prefill_batch != beam_size_) {
        return InferenceError::InvalidArgument;
    }
    const int32_t input_batch = prefill_batch == 0 ? 1 : prefill_batch;
    std::vector<int32_t> input_ids(static_cast<size_t>(input_batch) * causal_ids.size());
    for (int32_t batch = 0; batch < input_batch; ++batch) {
        std::copy(causal_ids.begin(), causal_ids.end(),
                  input_ids.begin() + static_cast<size_t>(batch) * causal_ids.size());
    }
    context::PersistentTensor& prefill_cache = input_batch == 1 ? fill_kv_ : self_kv_;

    const InferenceError error = engine_.run_pre_pass1(
        input_ids, prefill_cache, 0, input_batch);
    if (error != InferenceError::Ok) {
        return error;
    }
    if (input_batch == 1) {
        self_kv_.copy_batch_slice_from(
            fill_kv_, 0, 0, 0, static_cast<int32_t>(causal_ids.size()));
        self_kv_.copy_batch_slice_to_all(0, 0, static_cast<int32_t>(causal_ids.size()));
    }
    history_ids_ = effective_ids;
    history_seqlen_ = static_cast<int32_t>(causal_ids.size());
    current_seqlen_ = history_seqlen_;
    return InferenceError::Ok;
}

InferenceError InferenceSession::replace_context_impl(const std::vector<int32_t>& context_ids) {
    std::vector<int32_t> target = without_bos(context_ids);
    if (std::any_of(target.begin(), target.end(), [&](int32_t id) {
            return id < 0 || id >= engine_.tokenizer().context_vocab_size();
        })) {
        return InferenceError::InvalidArgument;
    }
    const int32_t capacity = history_capacity();
    const int32_t target_window = eviction_target();
    if (target.size() > static_cast<size_t>(capacity)) {
        target.erase(target.begin(), target.end() - target_window);
    }
    if (target == history_ids_) {
        current_seqlen_ = history_seqlen_;
        return InferenceError::Ok;
    }
    if (target.size() < history_ids_.size() &&
        std::equal(target.begin(), target.end(), history_ids_.begin())) {
        history_ids_.resize(target.size());
        history_seqlen_ = 1 + static_cast<int32_t>(target.size());
        current_seqlen_ = history_seqlen_;
        return InferenceError::Ok;
    }
    if (is_history_prefix(target) && history_seqlen_ > 0) {
        // Generated positions are temporary. The contiguous fill cache holds
        // the clean causal history, so appending can still update only the
        // new slice even when current_seqlen_ is ahead of history_seqlen_.
        return fill_incremental(std::vector<int32_t>(
            target.begin() + static_cast<std::ptrdiff_t>(history_ids_.size()), target.end()));
    }
    return fill_from_scratch(target);
}

InferenceError InferenceSession::fill_impl(const std::vector<int32_t>& new_ids) {
    const std::vector<int32_t> ids = without_bos(new_ids);
    if (std::any_of(ids.begin(), ids.end(), [&](int32_t id) {
            return id < 0 || id >= engine_.tokenizer().context_vocab_size();
        })) {
        return InferenceError::InvalidArgument;
    }
    if (ids.empty()) {
        return InferenceError::Ok;
    }
    if (history_seqlen_ == 0) {
        return fill_from_scratch(ids);
    }
    return fill_incremental(ids);
}

InferenceError InferenceSession::fill(context::Context& context,
                                      const std::vector<int32_t>& new_ids) {
    sync_from_context(context);
    const InferenceError error = fill_impl(new_ids);
    sync_to_context(context);
    return error;
}

InferenceError InferenceSession::replace_context(context::Context& context,
                                                 const std::vector<int32_t>& context_ids) {
    sync_from_context(context);
    const InferenceError error = replace_context_impl(context_ids);
    sync_to_context(context);
    return error;
}

bool InferenceSession::cancelled(CancellationFn cancellation, void* user_data) const {
    return cancellation != nullptr && cancellation(user_data);
}

void InferenceSession::sync_from_context(context::Context& context) {
    self_kv_ = context.self_kv;
    beam_size_ = self_kv_.batch_size();
    history_ids_.clear();
    if (context.context_ids_len() > 0) {
        history_ids_.assign(context.context_ids(),
                            context.context_ids() + context.context_ids_len());
    }
    current_seqlen_ = context.current_seqlen();
    history_seqlen_ = context.history_seqlen();
}

void InferenceSession::sync_to_context(context::Context& context) {
    if (history_ids_.size() > static_cast<size_t>(context.context_ids_capacity())) {
        throw std::out_of_range("InferenceSession: context slot capacity exceeded");
    }
    context.truncate_context_ids(0);
    if (!history_ids_.empty()) {
        context.append_context_ids(history_ids_.data(),
                                   static_cast<int32_t>(history_ids_.size()));
    }
    context.set_current_seqlen(current_seqlen_);
    context.set_history_seqlen(history_seqlen_);
}

GenerateResult InferenceSession::generate_impl(const std::vector<int32_t>& pinyin_ids,
                                               const std::vector<int32_t>& context_ids,
                                               CancellationFn cancellation,
                                               void* cancellation_user_data) {
    GenerateResult result;
    const auto refresh_state = [&]() {
        result.current_seqlen = current_seqlen_;
        result.history_seqlen = history_seqlen_;
    };
    refresh_state();

    if (!context_ids.empty()) {
        const InferenceError fill_error = replace_context_impl(context_ids);
        if (fill_error != InferenceError::Ok) {
            result.error = fill_error;
            refresh_state();
            return result;
        }
    }
    refresh_state();

    // Window the committed history before generation: once it reaches
    // max_history_length, evict down to max_history_length - slack_interval
    // (keeping the first BOS as the attention sink) so the incoming pinyin
    // window still fits below the pre-model cache limit.
    if (static_cast<int32_t>(history_ids_.size()) >= core_config_.max_history_length) {
        const InferenceError trunc_error = truncate_history(eviction_target());
        if (trunc_error != InferenceError::Ok) {
            result.error = trunc_error;
            refresh_state();
            return result;
        }
        refresh_state();
    }

    if (pinyin_ids.empty()) {
        result.beams.push_back(BeamResult{0.0, {}, {}});
        return result;
    }
    if (std::any_of(pinyin_ids.begin(), pinyin_ids.end(), [&](int32_t id) {
            return id < 0 || id >= engine_.tokenizer().pinyin_vocab_size();
        })) {
        result.error = InferenceError::InvalidArgument;
        return result;
    }

    // Any Cancelled return must leave the context cursors at the committed
    // history state so an interrupted generation can be retried from a
    // consistent position (the temporary generation positions are never
    // committed).
    const auto rollback_generation_cursor = [&]() {
        current_seqlen_ = history_seqlen_;
    };
    if (cancelled(cancellation, cancellation_user_data)) {
        rollback_generation_cursor();
        result.error = InferenceError::Cancelled;
        refresh_state();
        return result;
    }

    const auto& cfg = engine_.config();
    if (static_cast<int32_t>(pinyin_ids.size()) > core_config_.max_pinyin_length) {
        result.error = InferenceError::PinyinLimitExceeded;
        return result;
    }
    const bool has_history = history_seqlen_ > 0;
    if (!has_history) {
        // The first decoding step overwrites position zero and causal masks
        // hide every dirty future position, so stale cache data is harmless.
        current_seqlen_ = 0;
        refresh_state();
    }

    PostModelOutput post;
    InferenceError error = engine_.run_post_model(pinyin_ids, post);
    if (error != InferenceError::Ok) {
        result.error = error;
        refresh_state();
        return result;
    }

    const int32_t max_pre = cfg.pre_model.max_seqlen;
    const int32_t first_position = has_history ? history_seqlen_ - 1 : 0;
    const int32_t last_position = has_history
                                      ? history_seqlen_ + static_cast<int32_t>(pinyin_ids.size()) - 2
                                      : static_cast<int32_t>(pinyin_ids.size()) - 1;
    if (first_position < 0 || last_position >= max_pre) {
        result.error = InferenceError::ContextLimitExceeded;
        refresh_state();
        return result;
    }

    const int32_t pre2_batch = engine_.pre_pass2_batch_size();
    if (pre2_batch != 0 && pre2_batch != 1 && pre2_batch != beam_size_) {
        result.error = InferenceError::InvalidArgument;
        refresh_state();
        return result;
    }
    const int32_t first_batch = pre2_batch == beam_size_ ? beam_size_ : 1;
    std::vector<int32_t> first_input(static_cast<size_t>(first_batch),
                                     has_history ? history_ids_.back()
                                                 : engine_.tokenizer().special_token_id("bos_token"));
    std::vector<int32_t> first_positions(static_cast<size_t>(first_batch), first_position);
    context::PersistentTensor& first_cache = first_batch == 1 ? fill_kv_ : self_kv_;
    const std::vector<float> clean_history_last = has_history
                                                       ? first_cache.snapshot_batch_slice(0, first_position, 1)
                                                       : std::vector<float>();
    DecoderModelOutput decoder;
    error = engine_.run_pre_pass2(first_input, first_cache, first_positions, post, 0, decoder);
    if (has_history) {
        for (int32_t batch = 0; batch < first_batch; ++batch) {
            first_cache.restore_batch_slice(batch, first_position, 1, clean_history_last);
        }
    } else if (first_batch == 1) {
        self_kv_.copy_batch_slice_from(fill_kv_, 0, 0, 0, 1);
        self_kv_.copy_batch_slice_to_all(0, 0, 1);
    }
    refresh_state();
    if (cancelled(cancellation, cancellation_user_data)) {
        rollback_generation_cursor();
        result.error = InferenceError::Cancelled;
        refresh_state();
        return result;
    }
    if (error != InferenceError::Ok) {
        result.error = error;
        return result;
    }

    const std::vector<TokenScore> first_tokens = top_tokens(decoder.logits, 0,
                                                            decoder.projection_size, beam_size_);
    if (static_cast<int32_t>(first_tokens.size()) < beam_size_) {
        result.error = InferenceError::NoCandidates;
        refresh_state();
        return result;
    }

    std::vector<BeamResult> beams(static_cast<size_t>(beam_size_));
    for (int32_t beam = 0; beam < beam_size_; ++beam) {
        beams[static_cast<size_t>(beam)].score = first_tokens[static_cast<size_t>(beam)].log_prob;
        beams[static_cast<size_t>(beam)].pred_ids.push_back(
            decoder.candidate_ids[static_cast<size_t>(first_tokens[static_cast<size_t>(beam)].id)]);
    }
    current_seqlen_ = has_history ? history_seqlen_ : 1;
    refresh_state();

    for (int32_t step = 1; step < static_cast<int32_t>(pinyin_ids.size()); ++step) {
        if (cancelled(cancellation, cancellation_user_data)) {
            rollback_generation_cursor();
            result.error = InferenceError::Cancelled;
            result.beams = std::move(beams);
            refresh_state();
            return result;
        }

        std::vector<int32_t> input_ids(static_cast<size_t>(beam_size_));
        for (int32_t beam = 0; beam < beam_size_; ++beam) {
            const int32_t chinese_id = beams[static_cast<size_t>(beam)].pred_ids.back();
            input_ids[static_cast<size_t>(beam)] =
                engine_.tokenizer().chinese_id_to_context_id(chinese_id);
            if (input_ids[static_cast<size_t>(beam)] < 0) {
                rollback_generation_cursor();
                result.error = InferenceError::InvalidArgument;
                refresh_state();
                return result;
            }
        }
        const int32_t position = has_history ? history_seqlen_ + step - 1 : step;
        std::vector<int32_t> positions(static_cast<size_t>(beam_size_), position);
        error = engine_.run_pre_pass2(input_ids, self_kv_, positions, post, step, decoder);
        if (cancelled(cancellation, cancellation_user_data)) {
            rollback_generation_cursor();
            result.error = InferenceError::Cancelled;
            result.beams = std::move(beams);
            refresh_state();
            return result;
        }
        if (error != InferenceError::Ok) {
            rollback_generation_cursor();
            result.error = error;
            refresh_state();
            return result;
        }

        struct Expansion {
            double score;
            int32_t parent;
            int32_t token;
        };
        std::vector<Expansion> expansions;
        expansions.reserve(static_cast<size_t>(beam_size_) * beam_size_);
        for (int32_t parent = 0; parent < beam_size_; ++parent) {
            const auto tokens = top_tokens(
                decoder.logits, static_cast<size_t>(parent) * decoder.projection_size,
                decoder.projection_size, beam_size_);
            for (const auto& token : tokens) {
                expansions.push_back(Expansion{
                    beams[static_cast<size_t>(parent)].score + token.log_prob, parent,
                    decoder.candidate_ids[static_cast<size_t>(token.id)]});
            }
        }
        if (static_cast<int32_t>(expansions.size()) < beam_size_) {
            rollback_generation_cursor();
            result.error = InferenceError::NoCandidates;
            refresh_state();
            return result;
        }
        std::partial_sort(expansions.begin(), expansions.begin() + beam_size_, expansions.end(),
                          [](const Expansion& lhs, const Expansion& rhs) {
                              if (lhs.score != rhs.score) return lhs.score > rhs.score;
                              if (lhs.parent != rhs.parent) return lhs.parent < rhs.parent;
                              return lhs.token < rhs.token;
                          });

        std::vector<BeamResult> next_beams(static_cast<size_t>(beam_size_));
        std::vector<int32_t> parents(static_cast<size_t>(beam_size_));
        for (int32_t beam = 0; beam < beam_size_; ++beam) {
            const Expansion& expansion = expansions[static_cast<size_t>(beam)];
            parents[static_cast<size_t>(beam)] = expansion.parent;
            next_beams[static_cast<size_t>(beam)] = beams[static_cast<size_t>(expansion.parent)];
            next_beams[static_cast<size_t>(beam)].score = expansion.score;
            next_beams[static_cast<size_t>(beam)].pred_ids.push_back(expansion.token);
        }
        // Committed history is identical for every beam. Only the locally
        // causal generation suffix can differ and therefore needs reordering.
        const int32_t recursive_start = history_seqlen_;
        const int32_t recursive_length = position - recursive_start + 1;
        self_kv_.reorder_batch_slice(
            parents, recursive_start, recursive_length, reorder_scratch_);
        beams = std::move(next_beams);
        current_seqlen_ = position + 1;
        refresh_state();
    }

    for (auto& beam : beams) {
        beam.decoded = engine_.tokenizer().ids_to_text(beam.pred_ids);
    }
    result.beams = std::move(beams);
    result.current_seqlen = current_seqlen_;
    result.history_seqlen = history_seqlen_;
    return result;
}

GenerateResult InferenceSession::generate(context::Context& context,
                                          const std::vector<int32_t>& pinyin_ids,
                                          const std::vector<int32_t>& context_ids,
                                          CancellationFn cancellation,
                                          void* cancellation_user_data) {
    sync_from_context(context);
    GenerateResult result = generate_impl(pinyin_ids, context_ids, cancellation,
                                          cancellation_user_data);
    sync_to_context(context);
    return result;
}

}  // namespace phono::engine
