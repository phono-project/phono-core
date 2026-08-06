// context.hpp
//
// Context + ContextManager: the streaming-context state that used to live
// inside InferenceSession (KV caches + current cursor), factored out so a
// single session can drive several independent contexts and so all buffers
// are allocated up front in a few contiguous blocks.
//
//   ContextManager — owns the buffers and creates the Context objects:
//       * a STACKED pre-model self-KV cache   (num_contexts, pre_layers, 2, B, pre_max, ...)
//       * a STACKED shared cross-KV cache     (num_contexts, 2, B, pre_max, ...)
//       * a STACKED context-ids tensor        (num_contexts, max_ids) int32
//     Each Context gets a `PersistentTensor::view()` into the stacked KV
//     caches plus a pointer into the stacked ids buffer — no per-context
//     allocation, so no memory fragmentation. All buffers are zeroed at init.
//
//   Context — the persistent per-conversation state. Created once at manager
//   init, then passed by reference to InferenceSession's
//   initialize_context()/advance_context()/predict_step*(), which read its
//   key params and mutate it in place (write KV caches, append tokenizer
//   ids, advance current_position).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "context/kv_cache.hpp"

namespace phono::context {

class Context {
public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept = default;
    Context& operator=(Context&&) noexcept = default;

    // KV-cache views into the ContextManager's stacked buffers. The .pte
    // graphs write into these in place; the backing storage is shared with
    // the manager and with every other context's view.
    PersistentTensor pre_self_kv;
    PersistentTensor cross_kv;

    // Tokenizer-id sequence of the committed context text so far (BOS is NOT
    // stored here — see initialize_context). Stored in the manager's stacked
    // ids buffer; `context_ids()` points into it.
    const int32_t* context_ids() const { return ids_ptr_; }
    int32_t context_ids_len() const { return ids_len_; }
    int32_t context_ids_capacity() const { return ids_capacity_; }

    // Total number of pre-model tokens encoded into the KV caches so far
    // (BOS + every committed context char). This is the cache write position
    // / cross-attention KV length for the next step.
    int32_t current_position() const { return current_position_; }
    void set_current_position(int32_t p) { current_position_ = p; }

    bool empty() const { return ids_len_ == 0; }

    // Appends ids to the context's tokenizer-id sequence (throws
    // std::out_of_range if the stacked ids capacity would be exceeded).
    void append_context_ids(const int32_t* ids, int32_t n);
    void append_context_id(int32_t id) { append_context_ids(&id, 1); }

    // Resets this context to a fresh, unused state: zeroes its slice of the
    // stacked KV caches and clears the cursor + id sequence. The next
    // advance_context() will re-seed BOS.
    void clear();

private:
    friend class ContextManager;
    int32_t* ids_ptr_ = nullptr;   // points into ContextManager::stacked_ids_
    int32_t ids_len_ = 0;
    int32_t ids_capacity_ = 0;
    int32_t current_position_ = 0;
};

class ContextManager {
public:
    // Allocates the stacked KV caches and the stacked context-ids buffer up
    // front (one contiguous block per kind) and creates `num_contexts`
    // Context objects as views into them. Contexts are persistent and reused
    // for the manager's whole lifetime.
    ContextManager(const core::ModelPackageConfig& cfg, size_t num_contexts);

    size_t num_contexts() const { return contexts_.size(); }
    size_t context_ids_capacity() const { return max_ids_; }

    // Retrieval by slot index. Throws std::out_of_range for ids outside
    // [0, num_contexts).
    Context& get_context_by_id(int32_t id);
    const Context& get_context_by_id(int32_t id) const;

    // Retrieval by exact tokenizer-id sequence.
    Context* get_context_by_full_matching(const std::vector<int32_t>& token_ids);
    const Context* get_context_by_full_matching(const std::vector<int32_t>& token_ids) const;

    const core::ModelPackageConfig& config() const { return cfg_; }

private:
    Context make_context(size_t index);

    core::ModelPackageConfig cfg_;
    PersistentTensor stacked_pre_self_kv_;  // (num_contexts, pre_layers, 2, B, pre_max, ...)
    PersistentTensor stacked_cross_kv_;     // (num_contexts, 2, B, pre_max, ...)
    std::vector<int32_t> stacked_ids_;      // (num_contexts, max_ids) int32
    std::vector<Context> contexts_;         // fixed size, addresses stable after init
    size_t max_ids_ = 0;
};

}  // namespace phono::context
