#include "context/context.hpp"

#include <algorithm>
#include <stdexcept>

namespace phono::context {

void Context::append_context_ids(const int32_t* ids, int32_t n) {
    if (n < 0) {
        throw std::invalid_argument("Context::append_context_ids: negative count");
    }
    if (ids_len_ + n > ids_capacity_) {
        throw std::out_of_range(
            "Context::append_context_ids: context-id capacity exceeded (context text "
            "longer than the pre-model max_seqlen)");
    }
    for (int32_t i = 0; i < n; ++i) {
        ids_ptr_[ids_len_ + i] = ids[i];
    }
    ids_len_ += n;
}

void Context::truncate_context_ids(int32_t n) {
    if (n < 0 || n > ids_len_) {
        throw std::out_of_range("Context::truncate_context_ids: length out of bounds");
    }
    ids_len_ = n;
    current_position_ = std::min(current_position_, 1 + n);
    history_seqlen_ = std::min(history_seqlen_, 1 + n);
}

void Context::copy_cached_slice_to_all(int32_t start, int32_t length) {
    self_kv.copy_batch_slice_to_all(0, start, length);
}

void Context::shift_cached_tokens(int32_t discarded, int32_t retained) {
    for (int32_t batch = 0; batch < self_kv.batch_size(); ++batch) {
        self_kv.shift_batch_tokens(batch, discarded, retained);
    }
}

void Context::clear() {
    self_kv.zero_();
    ids_len_ = 0;
    current_position_ = 0;
    history_seqlen_ = 0;
}

ContextManager::ContextManager(const core::ModelPackageConfig& cfg, size_t num_contexts)
    : ContextManager(cfg, num_contexts, nlohmann::json::object()) {}

ContextManager::Params ContextManager::parse_params(const core::ModelPackageConfig& cfg,
                                                     const nlohmann::json& options) {
    if (!options.is_object()) {
        throw std::invalid_argument("ContextManager: options must be a JSON object");
    }

    Params params;
    params.beam_size = options.value("beam_size", cfg.runtime.batch_size);
    params.slack_tokens = options.value("N", params.slack_tokens);
    params.match_threshold = options.value("T", params.match_threshold);
    params.max_context_length = options.value("max_context_length", cfg.pre_model.max_seqlen - 1);

    if (params.beam_size <= 0) {
        throw std::invalid_argument("ContextManager: beam_size must be positive");
    }
    if (params.slack_tokens < 0) {
        throw std::invalid_argument("ContextManager: N must be non-negative");
    }
    if (params.match_threshold <= 0) {
        throw std::invalid_argument("ContextManager: T must be positive");
    }
    if (params.max_context_length < 0 || params.max_context_length + 1 > cfg.pre_model.max_seqlen) {
        throw std::invalid_argument(
            "ContextManager: max_context_length must fit pre_model.max_seqlen including BOS");
    }
    return params;
}

ContextManager::ContextManager(const core::ModelPackageConfig& cfg, size_t num_contexts,
                               const nlohmann::json& options)
    : cfg_(cfg), params_(parse_params(cfg, options)) {
    if (num_contexts == 0) {
        throw std::invalid_argument("ContextManager: num_contexts must be > 0");
    }
    const int32_t B = params_.beam_size;
    const int32_t pre_max = cfg.pre_model.max_seqlen;
    const int32_t n = static_cast<int32_t>(num_contexts);

    // One contiguous zeroed allocation with the context index as the leading
    // dimension. Every Context is a view into it.
    stacked_pre_self_kv_ = make_zero_persistent_tensor(
        {n, cfg.pre_model.mhsa_layers, 2, B, pre_max, cfg.pre_model.mhsa_heads,
         cfg.pre_model.self_head_dim()});

    max_ids_ = static_cast<size_t>(params_.max_context_length);
    stacked_ids_.assign(num_contexts * max_ids_, 0);
    use_ticks_.assign(num_contexts, 0);

    contexts_.reserve(num_contexts);
    for (size_t i = 0; i < num_contexts; ++i) {
        contexts_.push_back(make_context(i));
    }
}

Context ContextManager::make_context(size_t index) {
    const int32_t B = params_.beam_size;
    const int32_t pre_max = cfg_.pre_model.max_seqlen;
    const int64_t pre_layers = cfg_.pre_model.mhsa_layers;
    const int64_t pre_heads = cfg_.pre_model.mhsa_heads;
    const int64_t pre_hd = cfg_.pre_model.self_head_dim();
    const int64_t pre_stride = pre_layers * 2 * B * pre_max * pre_heads * pre_hd;
    Context c;
    c.self_kv = stacked_pre_self_kv_.view(
        static_cast<int64_t>(index) * pre_stride,
        {cfg_.pre_model.mhsa_layers, 2, B, pre_max, cfg_.pre_model.mhsa_heads,
         cfg_.pre_model.self_head_dim()});
    c.ids_ptr_ = stacked_ids_.data() + static_cast<int64_t>(index) * static_cast<int64_t>(max_ids_);
    c.ids_capacity_ = static_cast<int32_t>(max_ids_);
    return c;
}

Context& ContextManager::get_context_by_id(int32_t id) {
    if (id < 0 || static_cast<size_t>(id) >= contexts_.size()) {
        throw std::out_of_range("ContextManager::get_context_by_id: id out of range");
    }
    return contexts_[static_cast<size_t>(id)];
}

const Context& ContextManager::get_context_by_id(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= contexts_.size()) {
        throw std::out_of_range("ContextManager::get_context_by_id: id out of range");
    }
    return contexts_[static_cast<size_t>(id)];
}

Context* ContextManager::get_context_by_full_matching(const std::vector<int32_t>& token_ids) {
    for (auto& ctx : contexts_) {
        const int32_t len = ctx.context_ids_len();
        if (len == static_cast<int32_t>(token_ids.size()) &&
            (token_ids.empty() ||
             std::equal(ctx.context_ids(), ctx.context_ids() + len, token_ids.begin()))) {
            return &ctx;
        }
    }
    return nullptr;
}

const Context* ContextManager::get_context_by_full_matching(
    const std::vector<int32_t>& token_ids) const {
    for (const auto& ctx : contexts_) {
        const int32_t len = ctx.context_ids_len();
        if (len == static_cast<int32_t>(token_ids.size()) &&
            (token_ids.empty() ||
             std::equal(ctx.context_ids(), ctx.context_ids() + len, token_ids.begin()))) {
            return &ctx;
        }
    }
    return nullptr;
}

size_t ContextManager::choose_replacement_slot() {
    for (size_t i = 0; i < contexts_.size(); ++i) {
        if (contexts_[i].empty()) {
            return i;
        }
    }

    return static_cast<size_t>(std::min_element(use_ticks_.begin(), use_ticks_.end()) -
                               use_ticks_.begin());
}

Context* ContextManager::get_context_auto(const std::vector<int32_t>& token_ids) {
    if (token_ids.size() > max_ids_) {
        throw std::out_of_range("ContextManager::get_context_auto: context length exceeds configured limit");
    }

    const int32_t target_len = static_cast<int32_t>(token_ids.size());
    int best_prefix = -1;
    int best_prefix_len = -1;
    for (size_t i = 0; i < contexts_.size(); ++i) {
        const Context& ctx = contexts_[i];
        const int32_t slot_len = ctx.context_ids_len();
        if (slot_len == 0 && target_len != 0) {
            continue;
        }
        const int32_t prefix_length = std::min(slot_len, target_len);
        int32_t common = 0;
        while (common < prefix_length &&
               ctx.context_ids()[common] == token_ids[static_cast<size_t>(common)]) {
            ++common;
        }
        if (common > 0 && common > best_prefix_len) {
            best_prefix = static_cast<int>(i);
            best_prefix_len = common;
        }
    }

    struct Match {
        int slot = -1;
        int offset = 0;
        int length = 0;
    } best;

    for (size_t i = 0; i < contexts_.size(); ++i) {
        const Context& ctx = contexts_[i];
        const int32_t slot_len = ctx.context_ids_len();
        for (int offset = 1; offset < slot_len; ++offset) {
            const int length = std::min(target_len, slot_len - offset);
            int common = 0;
            while (common < length &&
                   ctx.context_ids()[offset + common] == token_ids[static_cast<size_t>(common)]) {
                ++common;
            }
            if (common >= params_.match_threshold &&
                (common > best.length || (common == best.length && offset < best.offset))) {
                best = Match{static_cast<int>(i), offset, common};
            }
        }
    }

    // Prefer a larger slack shift when it preserves a reusable match. This
    // avoids repeatedly moving one token in a sliding input window.
    if (params_.slack_tokens > 1 && best.slot >= 0 && best.offset < params_.slack_tokens) {
        for (size_t i = 0; i < contexts_.size(); ++i) {
            const Context& ctx = contexts_[i];
            const int32_t slot_len = ctx.context_ids_len();
            for (int offset = params_.slack_tokens; offset < slot_len; ++offset) {
                const int length = std::min(target_len, slot_len - offset);
                int common = 0;
                while (common < length &&
                       ctx.context_ids()[offset + common] == token_ids[static_cast<size_t>(common)]) {
                    ++common;
                }
                if (common >= params_.match_threshold && common >= best.length) {
                    best = Match{static_cast<int>(i), offset, common};
                }
            }
        }
    }

    if (best_prefix >= 0 && best_prefix_len >= best.length) {
        Context& ctx = contexts_[static_cast<size_t>(best_prefix)];
        ctx.truncate_context_ids(best_prefix_len);
        ctx.set_history_seqlen(1 + best_prefix_len);
        ctx.set_current_position(1 + best_prefix_len);
        use_ticks_[static_cast<size_t>(best_prefix)] = next_tick_++;
        return &ctx;
    }

    size_t selected;
    if (best.slot >= 0) {
        selected = static_cast<size_t>(best.slot);
        Context& ctx = contexts_[selected];
        ctx.shift_cached_tokens(best.offset, best.length);
        ctx.truncate_context_ids(0);
        ctx.append_context_ids(token_ids.data(), best.length);
        ctx.set_history_seqlen(1 + best.length);
        ctx.set_current_position(1 + best.length);
    } else {
        selected = choose_replacement_slot();
        contexts_[selected].clear();
    }

    use_ticks_[selected] = next_tick_++;
    return &contexts_[selected];
}

}  // namespace phono::context
