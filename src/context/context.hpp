// context.hpp
//
// Context + ContextManager: the streaming-context state that used to live
// inside InferenceSession (KV caches + current cursor), factored out so a
// single session can drive several independent contexts and so all buffers
// are allocated up front in a few contiguous blocks.
//
//   ContextManager — owns the buffers and creates the Context objects:
//       * a STACKED pre-model self-KV cache   (num_contexts, pre_layers, 2, B, pre_max, ...)
//       * a STACKED context-ids tensor        (num_contexts, max_ids) int32
//     Each Context gets a `PersistentTensor::view()` into the stacked KV
//     caches plus a pointer into the stacked ids buffer — no per-context
//     allocation, so no memory fragmentation. All buffers are zeroed at init.
//
//   Context — the persistent cached context state. Created once at manager
//   init and reused by get_context_auto().
#pragma once

#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
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

    // KV-cache view into the ContextManager's stacked buffer. The .pte graph
    // writes it in place; the B dimension is batch_size / beam_size.
    PersistentTensor self_kv;

    // Tokenizer-id sequence of the cached context text (BOS is not stored).
    // The pointer refers to the manager's stacked ids buffer.
    const int32_t* context_ids() const { return ids_ptr_; }
    int32_t context_ids_len() const { return ids_len_; }
    int32_t context_ids_capacity() const { return ids_capacity_; }

    // Number of causal positions currently valid in the cache, including BOS.
    int32_t current_position() const { return current_position_; }
    void set_current_position(int32_t p) { current_position_ = p; }
    int32_t current_seqlen() const { return current_position_; }
    void set_current_seqlen(int32_t p) { current_position_ = p; }

    int32_t history_seqlen() const { return history_seqlen_; }
    void set_history_seqlen(int32_t p) { history_seqlen_ = p; }

    bool empty() const { return ids_len_ == 0; }

    // Appends ids to the cached context-id sequence.
    void append_context_ids(const int32_t* ids, int32_t n);
    void append_context_id(int32_t id) { append_context_ids(&id, 1); }

    void truncate_context_ids(int32_t n);

    // Cache operations used by ContextManager's automatic reuse path.
    void copy_cached_slice_to_all(int32_t start, int32_t length);
    void shift_cached_tokens(int32_t discarded, int32_t retained);

    // Resets this context to a fresh, unused state: zeroes its slice of the
    // stacked KV cache and clears the sequence metadata.
    void clear();

private:
    friend class ContextManager;
    int32_t* ids_ptr_ = nullptr;   // points into ContextManager::stacked_ids_
    int32_t ids_len_ = 0;
    int32_t ids_capacity_ = 0;
    int32_t current_position_ = 0;
    int32_t history_seqlen_ = 0;
};

class ContextManager {
public:
    using Slot = Context;

    struct Params {
        int32_t beam_size = 1;
        int32_t slack_tokens = 8;    // JSON key: N
        int32_t match_threshold = 8; // JSON key: T
        int32_t max_context_length = 127;
        double trial_ratio = 0.25;
        double decay_alpha = 0.5;
        double decay_lambda = 1.0 / 60.0;
    };

    // Allocates the stacked KV caches and the stacked context-ids buffer up
    // front (one contiguous block per kind) and creates `num_contexts`
    // Context objects as views into them. Contexts are persistent and reused
    // for the manager's whole lifetime.
    ContextManager(const core::ModelPackageConfig& cfg, size_t num_contexts);
    ContextManager(const core::ModelPackageConfig& cfg, size_t num_contexts,
                   const nlohmann::json& options);
    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;

    size_t num_contexts() const { return contexts_.size(); }
    size_t context_ids_capacity() const { return max_ids_; }
    const Params& params() const { return params_; }

    // Retrieval by slot index. Throws std::out_of_range for ids outside
    // [0, num_contexts).
    Context& get_context_by_id(int32_t id);
    const Context& get_context_by_id(int32_t id) const;

    // Retrieval by exact tokenizer-id sequence.
    Context* get_context_by_full_matching(const std::vector<int32_t>& token_ids);
    const Context* get_context_by_full_matching(const std::vector<int32_t>& token_ids) const;

    // Selects the best reusable slot for `token_ids` and updates its cache
    // metadata in place. The returned slot contains the longest reusable
    // prefix or middle substring; callers append/fill token_ids from
    // context_ids_len() onward. Input ids exclude BOS.
    Context* get_context_auto(const std::vector<int32_t>& token_ids);

    const core::ModelPackageConfig& config() const { return cfg_; }

private:
    static Params parse_params(const core::ModelPackageConfig& cfg,
                               const nlohmann::json& options);
    Context make_context(size_t index);
    size_t choose_replacement_slot();
    double slot_value(size_t index) const;
    void touch_slot(size_t index, bool promote);
    size_t weakest_slot(bool trial_only, size_t excluded) const;
    size_t trial_capacity() const;

    core::ModelPackageConfig cfg_;
    Params params_;
    PersistentTensor stacked_pre_self_kv_;  // (num_contexts, pre_layers, 2, B, pre_max, ...)
    std::vector<int32_t> stacked_ids_;      // (num_contexts, max_ids) int32
    std::vector<Context> contexts_;         // fixed size, addresses stable after init
    std::vector<bool> protected_slots_;
    std::vector<std::chrono::steady_clock::time_point> last_access_;
    size_t max_ids_ = 0;
};

}  // namespace phono::context
