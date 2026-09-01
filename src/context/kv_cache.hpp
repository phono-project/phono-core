// kv_cache.hpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <executorch/extension/tensor/tensor.h>

#include "core/config.hpp"

namespace phono::context {

// Number of elements described by a shape (row-major product).
inline int64_t persistent_tensor_numel(const std::vector<int32_t>& shape) {
    int64_t n = 1;
    for (int32_t d : shape) n *= static_cast<int64_t>(d);
    return n;
}

// Owns one persistent, zero-initializable cache buffer: contiguous float32
// storage plus a TensorPtr view over it that is created once and reused for
// the buffer's whole lifetime. `storage` is a shared_ptr so that `view()`
// sub-tensors can alias the exact same backing memory without copying — the
// ContextManager's stacked buffer is the only place a KV cache is actually
// allocated.
struct PersistentTensor {
    std::shared_ptr<std::vector<float_t>> storage;  // shared so views alias the same memory
    std::vector<int32_t> shape;
    executorch::extension::TensorPtr tensor;  // non-owning view over `storage`
    int64_t storage_offset_elements_ = 0;     // absolute offset into `storage`

    // Zeroes this tensor's own slice of the storage in place (the whole
    // storage for a root tensor, or just the viewed range for a sub-view).
    // Same memory the TensorPtr already points at — no need to recreate it.
    void zero_() {
        if (!storage || storage->empty()) return;
        const int64_t numel = persistent_tensor_numel(shape);
        float_t* begin = storage->data() + storage_offset_elements_;
        std::fill(begin, begin + numel, 0.0f);
    }

    float_t* data() {
        return storage ? storage->data() + storage_offset_elements_ : nullptr;
    }

    const float_t* data() const {
        return storage ? storage->data() + storage_offset_elements_ : nullptr;
    }

    // Returns a sub-tensor view into this tensor's storage at the given
    // element offset with the given (usually smaller) shape. The view shares
    // the underlying storage (no copy) and gets a fresh TensorPtr over
    // `storage->data() + offset`. This is what lets a ContextManager stack
    // many contexts' KV caches into one contiguous allocation and hand each
    // context a per-slot view without any per-context allocations.
    PersistentTensor view(int64_t offset_elements, const std::vector<int32_t>& view_shape) const;

    // The cache layout is [layers, K/V, B, max_seqlen, heads, head_dim].
    // These helpers operate on token slices without copying unrelated history.
    int32_t batch_size() const;
    int32_t max_seqlen() const;
    int64_t token_stride_elements() const;
    int64_t batch_stride_elements() const;
    void copy_batch_slice_from(const PersistentTensor& source, int32_t source_batch,
                               int32_t dst_batch, int32_t start, int32_t length);
    void copy_batch_slice(int32_t src_batch, int32_t dst_batch,
                          int32_t start, int32_t length);
    void copy_batch_slice_to_all(int32_t src_batch, int32_t start, int32_t length);
    std::vector<float_t> snapshot_batch_slice(int32_t batch, int32_t start,
                                              int32_t length) const;
    void restore_batch_slice(int32_t batch, int32_t start, int32_t length,
                             const std::vector<float_t>& values);
    void shift_batch_tokens(int32_t batch, int32_t discarded, int32_t retained);
    void reorder_batch_slice(const std::vector<int32_t>& parent_batches,
                             int32_t start, int32_t length,
                             std::vector<float_t>& scratch);
    void reorder_batches(const std::vector<int32_t>& parent_batches);
};

// Allocates a zero-initialized PersistentTensor with the given shape. The
// returned tensor owns a single contiguous float32 buffer.
PersistentTensor make_zero_persistent_tensor(const std::vector<int32_t>& shape);

}  // namespace phono::context
