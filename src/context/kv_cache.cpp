#include "context/kv_cache.hpp"

#include <stdexcept>

namespace phono::context {

using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;

PersistentTensor make_zero_persistent_tensor(const std::vector<int32_t>& shape) {
    PersistentTensor pt;
    pt.shape = shape;
    // Non-owning view: `make_tensor_ptr(sizes, raw_ptr, dtype)` wraps
    // `pt.storage->data()` without copying. `pt.storage` must outlive
    // `pt.tensor`; the shared_ptr keeps the vector alive as long as any root
    // PersistentTensor (or a `view()` derived from it) still references it.
    pt.storage = std::make_shared<std::vector<float_t>>(
        static_cast<size_t>(persistent_tensor_numel(shape)), 0.0f);
    pt.storage_offset_elements_ = 0;
    pt.tensor = make_tensor_ptr(pt.shape, pt.storage->data(), ScalarType::Float);
    return pt;
}

PersistentTensor PersistentTensor::view(int64_t offset_elements,
                                        const std::vector<int32_t>& view_shape) const {
    if (!storage) {
        throw std::runtime_error("PersistentTensor::view: parent has no storage");
    }
    const int64_t view_numel = persistent_tensor_numel(view_shape);
    const int64_t absolute = storage_offset_elements_ + offset_elements;
    if (absolute < 0 || absolute + view_numel > static_cast<int64_t>(storage->size())) {
        throw std::runtime_error("PersistentTensor::view: view out of bounds");
    }

    PersistentTensor v;
    v.storage = storage;              // share the exact same backing buffer
    v.storage_offset_elements_ = absolute;
    v.shape = view_shape;
    v.tensor = make_tensor_ptr(view_shape, storage->data() + absolute, ScalarType::Float);
    return v;
}

namespace {

void check_cache_shape(const PersistentTensor& cache) {
    if (cache.shape.size() != 6) {
        throw std::invalid_argument(
            "PersistentTensor: cache operations require [layers, 2, B, max, heads, head_dim]");
    }
    if (cache.shape[1] != 2 || cache.shape[2] <= 0 || cache.shape[3] <= 0) {
        throw std::invalid_argument("PersistentTensor: invalid KV-cache shape");
    }
}

}  // namespace

int32_t PersistentTensor::batch_size() const {
    check_cache_shape(*this);
    return shape[2];
}

int32_t PersistentTensor::max_seqlen() const {
    check_cache_shape(*this);
    return shape[3];
}

int64_t PersistentTensor::token_stride_elements() const {
    check_cache_shape(*this);
    return static_cast<int64_t>(shape[4]) * shape[5];
}

int64_t PersistentTensor::batch_stride_elements() const {
    check_cache_shape(*this);
    return static_cast<int64_t>(shape[3]) * token_stride_elements();
}

void PersistentTensor::copy_batch_slice_from(const PersistentTensor& source,
                                             int32_t source_batch, int32_t dst_batch,
                                             int32_t start, int32_t length) {
    check_cache_shape(*this);
    check_cache_shape(source);
    if (shape[0] != source.shape[0] || shape[3] != source.shape[3] ||
        shape[4] != source.shape[4] || shape[5] != source.shape[5] ||
        source_batch < 0 || source_batch >= source.shape[2] || dst_batch < 0 ||
        dst_batch >= shape[2] || start < 0 || length < 0 || start + length > shape[3]) {
        throw std::out_of_range("PersistentTensor::copy_batch_slice_from: range out of bounds");
    }

    const int64_t destination_token_stride = token_stride_elements();
    const int64_t destination_batch_stride = batch_stride_elements();
    const int64_t source_token_stride = source.token_stride_elements();
    const int64_t source_batch_stride = source.batch_stride_elements();
    const size_t bytes = static_cast<size_t>(length * destination_token_stride) * sizeof(float_t);
    for (int32_t layer = 0; layer < shape[0]; ++layer) {
        for (int32_t kv = 0; kv < 2; ++kv) {
            const int64_t destination_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * shape[2] * destination_batch_stride +
                static_cast<int64_t>(dst_batch) * destination_batch_stride +
                static_cast<int64_t>(start) * destination_token_stride;
            const int64_t source_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * source.shape[2] * source_batch_stride +
                static_cast<int64_t>(source_batch) * source_batch_stride +
                static_cast<int64_t>(start) * source_token_stride;
            std::memmove(data() + destination_offset, source.data() + source_offset, bytes);
        }
    }
}

void PersistentTensor::copy_batch_slice(int32_t src_batch, int32_t dst_batch,
                                        int32_t start, int32_t length) {
    check_cache_shape(*this);
    if (src_batch < 0 || dst_batch < 0 || src_batch >= shape[2] || dst_batch >= shape[2] ||
        start < 0 || length < 0 || start + length > shape[3]) {
        throw std::out_of_range("PersistentTensor::copy_batch_slice: range out of bounds");
    }
    const int64_t token_stride = token_stride_elements();
    const int64_t batch_stride = batch_stride_elements();
    const size_t bytes = static_cast<size_t>(length * token_stride) * sizeof(float_t);
    for (int32_t layer = 0; layer < shape[0]; ++layer) {
        for (int32_t kv = 0; kv < 2; ++kv) {
            const int64_t layer_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * shape[2] * batch_stride;
            const int64_t src_offset = layer_offset + static_cast<int64_t>(src_batch) * batch_stride +
                                       static_cast<int64_t>(start) * token_stride;
            const int64_t dst_offset = layer_offset + static_cast<int64_t>(dst_batch) * batch_stride +
                                       static_cast<int64_t>(start) * token_stride;
            std::memmove(data() + dst_offset, data() + src_offset, bytes);
        }
    }
}

void PersistentTensor::copy_batch_slice_to_all(int32_t src_batch, int32_t start,
                                                int32_t length) {
    check_cache_shape(*this);
    for (int32_t batch = 0; batch < shape[2]; ++batch) {
        if (batch != src_batch) {
            copy_batch_slice(src_batch, batch, start, length);
        }
    }
}

std::vector<float_t> PersistentTensor::snapshot_batch_slice(int32_t batch, int32_t start,
                                                            int32_t length) const {
    check_cache_shape(*this);
    if (batch < 0 || batch >= shape[2] || start < 0 || length < 0 || start + length > shape[3]) {
        throw std::out_of_range("PersistentTensor::snapshot_batch_slice: range out of bounds");
    }
    const int64_t token_stride = token_stride_elements();
    const int64_t batch_stride = batch_stride_elements();
    const size_t slice_elements = static_cast<size_t>(shape[0]) * 2 * length * token_stride;
    std::vector<float_t> values(slice_elements);
    size_t cursor = 0;
    for (int32_t layer = 0; layer < shape[0]; ++layer) {
        for (int32_t kv = 0; kv < 2; ++kv) {
            const int64_t layer_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * shape[2] * batch_stride;
            const int64_t offset = layer_offset + static_cast<int64_t>(batch) * batch_stride +
                                   static_cast<int64_t>(start) * token_stride;
            const size_t count = static_cast<size_t>(length * token_stride);
            std::memcpy(values.data() + cursor, data() + offset,
                        count * sizeof(float_t));
            cursor += count;
        }
    }
    return values;
}

void PersistentTensor::restore_batch_slice(int32_t batch, int32_t start, int32_t length,
                                            const std::vector<float_t>& values) {
    check_cache_shape(*this);
    if (batch < 0 || batch >= shape[2] || start < 0 || length < 0 || start + length > shape[3]) {
        throw std::out_of_range("PersistentTensor::restore_batch_slice: range out of bounds");
    }
    const int64_t token_stride = token_stride_elements();
    const int64_t batch_stride = batch_stride_elements();
    const size_t count = static_cast<size_t>(length * token_stride);
    const size_t expected = static_cast<size_t>(shape[0]) * 2 * count;
    if (values.size() != expected) {
        throw std::invalid_argument("PersistentTensor::restore_batch_slice: wrong value count");
    }
    size_t cursor = 0;
    for (int32_t layer = 0; layer < shape[0]; ++layer) {
        for (int32_t kv = 0; kv < 2; ++kv) {
            const int64_t layer_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * shape[2] * batch_stride;
            const int64_t offset = layer_offset + static_cast<int64_t>(batch) * batch_stride +
                                   static_cast<int64_t>(start) * token_stride;
            std::memcpy(data() + offset, values.data() + cursor,
                        count * sizeof(float_t));
            cursor += count;
        }
    }
}

void PersistentTensor::shift_batch_tokens(int32_t batch, int32_t discarded,
                                           int32_t retained) {
    check_cache_shape(*this);
    if (batch < 0 || batch >= shape[2] || discarded < 0 || retained < 0 ||
        discarded + retained > shape[3] - 1) {
        throw std::out_of_range("PersistentTensor::shift_batch_tokens: range out of bounds");
    }
    const int64_t token_stride = token_stride_elements();
    const int64_t batch_stride = batch_stride_elements();
    const size_t bytes = static_cast<size_t>(retained * token_stride) * sizeof(float_t);
    for (int32_t layer = 0; layer < shape[0]; ++layer) {
        for (int32_t kv = 0; kv < 2; ++kv) {
            const int64_t layer_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * shape[2] * batch_stride;
            float_t* base = data() + layer_offset + static_cast<int64_t>(batch) * batch_stride;
            std::memmove(base + token_stride, base + (1 + discarded) * token_stride, bytes);
        }
    }
}

void PersistentTensor::reorder_batch_slice(const std::vector<int32_t>& parent_batches,
                                           int32_t start, int32_t length,
                                           std::vector<float_t>& scratch) {
    check_cache_shape(*this);
    if (parent_batches.size() != static_cast<size_t>(shape[2])) {
        throw std::invalid_argument("PersistentTensor::reorder_batch_slice: wrong batch count");
    }
    for (int32_t parent : parent_batches) {
        if (parent < 0 || parent >= shape[2]) {
            throw std::out_of_range("PersistentTensor::reorder_batch_slice: parent out of bounds");
        }
    }
    if (start < 0 || length < 0 || start + length > shape[3]) {
        throw std::out_of_range("PersistentTensor::reorder_batch_slice: range out of bounds");
    }
    if (length == 0) return;

    const int64_t token_stride = token_stride_elements();
    const int64_t batch_stride = batch_stride_elements();
    const size_t slice_elements = static_cast<size_t>(length * token_stride);
    const size_t slice_bytes = slice_elements * sizeof(float_t);
    scratch.resize(parent_batches.size() * slice_elements);
    for (int32_t layer = 0; layer < shape[0]; ++layer) {
        for (int32_t kv = 0; kv < 2; ++kv) {
            const int64_t layer_offset =
                (static_cast<int64_t>(layer) * 2 + kv) * shape[2] * batch_stride;
            for (size_t dst = 0; dst < parent_batches.size(); ++dst) {
                const float_t* src = data() + layer_offset +
                                     static_cast<int64_t>(parent_batches[dst]) * batch_stride +
                                     static_cast<int64_t>(start) * token_stride;
                std::memcpy(scratch.data() + dst * slice_elements, src, slice_bytes);
            }
            for (size_t dst = 0; dst < parent_batches.size(); ++dst) {
                float_t* target = data() + layer_offset +
                                  static_cast<int64_t>(dst) * batch_stride +
                                  static_cast<int64_t>(start) * token_stride;
                std::memcpy(target, scratch.data() + dst * slice_elements, slice_bytes);
            }
        }
    }
}

void PersistentTensor::reorder_batches(const std::vector<int32_t>& parent_batches) {
    std::vector<float_t> scratch;
    reorder_batch_slice(parent_batches, 0, max_seqlen(), scratch);
}

}  // namespace phono::context
