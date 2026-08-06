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

void Context::clear() {
    pre_self_kv.zero_();  // zeroes only this context's slice.
    cross_kv.zero_();
    ids_len_ = 0;
    current_position_ = 0;
}

ContextManager::ContextManager(const core::ModelPackageConfig& cfg, size_t num_contexts) : cfg_(cfg) {
    if (num_contexts == 0) {
        throw std::invalid_argument("ContextManager: num_contexts must be > 0");
    }
    const int32_t B = cfg.runtime.batch_size;
    const int32_t pre_max = cfg.pre_model.max_seqlen;
    const int32_t n = static_cast<int32_t>(num_contexts);

    // One contiguous zeroed allocation per cache kind, with the context
    // index as the leading dimension. Every Context is a view into these.
    stacked_pre_self_kv_ = make_zero_persistent_tensor(
        {n, cfg.pre_model.mhsa_layers, 2, B, pre_max, cfg.pre_model.mhsa_heads,
         cfg.pre_model.self_head_dim()});
    stacked_cross_kv_ = make_zero_persistent_tensor(
        {n, 2, B, pre_max, cfg.post_model.mhca_heads, cfg.post_model.cross_head_dim()});

    // A context can hold at most pre_max tokens in the pre model (BOS + up to
    // pre_max-1 committed chars), so that is the per-context ids capacity.
    max_ids_ = static_cast<size_t>(pre_max);
    stacked_ids_.assign(num_contexts * max_ids_, 0);

    contexts_.reserve(num_contexts);
    for (size_t i = 0; i < num_contexts; ++i) {
        contexts_.push_back(make_context(i));
    }
}

Context ContextManager::make_context(size_t index) {
    const int32_t B = cfg_.runtime.batch_size;
    const int32_t pre_max = cfg_.pre_model.max_seqlen;
    const int64_t pre_layers = cfg_.pre_model.mhsa_layers;
    const int64_t pre_heads = cfg_.pre_model.mhsa_heads;
    const int64_t pre_hd = cfg_.pre_model.self_head_dim();
    const int64_t post_heads = cfg_.post_model.mhca_heads;
    const int64_t post_hd = cfg_.post_model.cross_head_dim();

    const int64_t pre_stride = pre_layers * 2 * B * pre_max * pre_heads * pre_hd;
    const int64_t cross_stride = 2 * B * pre_max * post_heads * post_hd;

    Context c;
    c.pre_self_kv = stacked_pre_self_kv_.view(
        static_cast<int64_t>(index) * pre_stride,
        {cfg_.pre_model.mhsa_layers, 2, B, pre_max, cfg_.pre_model.mhsa_heads,
         cfg_.pre_model.self_head_dim()});
    c.cross_kv = stacked_cross_kv_.view(
        static_cast<int64_t>(index) * cross_stride,
        {2, B, pre_max, cfg_.post_model.mhca_heads, cfg_.post_model.cross_head_dim()});
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

}  // namespace phono::context
