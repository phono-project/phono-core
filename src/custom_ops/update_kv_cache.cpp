#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <cstring>

namespace phono::ops {

using torch::executor::Tensor;
using torch::executor::RuntimeContext;

Tensor& update_kv_cache_out(
    RuntimeContext& ctx,
    const Tensor& cache,
    const Tensor& value,
    int64_t start_pos,
    Tensor& out) {
    
    int64_t batch_size = value.size(0);
    int64_t val_seq_len = value.size(1);
    int64_t out_seq_len = cache.size(1);

    int64_t inner_elements = 1;
    for (size_t i = 2; i < value.dim(); ++i) {
        inner_elements *= value.size(i);
    }

    size_t element_size = value.element_size();
    size_t inner_bytes = inner_elements * element_size;

    const char* src_ptr = value.const_data_ptr<char>();
    
    char* dst_ptr = const_cast<Tensor&>(cache).mutable_data_ptr<char>();

    size_t copy_bytes_per_batch = val_seq_len * inner_bytes;

    for (int64_t b = 0; b < batch_size; ++b) {
        const char* src_batch = src_ptr + (b * val_seq_len) * inner_bytes;
        char* dst_batch = dst_ptr + (b * out_seq_len + start_pos) * inner_bytes;
        std::memcpy(dst_batch, src_batch, copy_bytes_per_batch);
    }

    return out;
}

static void copy_kv_slice(
    const char* src_ptr,
    char* dst_ptr,
    int64_t batch_size,
    int64_t val_seq_len,
    int64_t out_seq_len,
    int64_t start_pos,
    int64_t head_stride_elements,
    size_t element_size)
{
    int64_t inner_elements = head_stride_elements;
    size_t inner_bytes = inner_elements * element_size;
    size_t copy_bytes_per_batch = val_seq_len * inner_bytes;

    for (int64_t b = 0; b < batch_size; ++b) {
        const char* src_batch = src_ptr + (b * val_seq_len) * inner_bytes;
        char* dst_batch = dst_ptr + (b * out_seq_len + start_pos) * inner_bytes;
        std::memcpy(dst_batch, src_batch, copy_bytes_per_batch);
    }
}

Tensor& update_cross_kv_out(
    RuntimeContext& ctx,
    const Tensor& cache,
    const Tensor& pre_K,
    const Tensor& pre_V,
    int64_t start_pos,
    Tensor& out)
{
    int64_t B = cache.size(1);
    int64_t maxlen = cache.size(2);

    int64_t head_elements = 1;
    for (size_t i = 3; i < cache.dim(); ++i) {
        head_elements *= cache.size(i);
    }
    size_t element_size = cache.element_size();

    int64_t channel_stride = B * maxlen * head_elements;
    char* dst_ptr = const_cast<Tensor&>(cache).mutable_data_ptr<char>();

    copy_kv_slice(
        pre_K.const_data_ptr<char>(),
        dst_ptr,
        pre_K.size(0), pre_K.size(1), maxlen, start_pos,
        head_elements, element_size);

    copy_kv_slice(
        pre_V.const_data_ptr<char>(),
        dst_ptr + channel_stride * element_size,
        pre_V.size(0), pre_V.size(1), maxlen, start_pos,
        head_elements, element_size);

    if (out.const_data_ptr() != cache.const_data_ptr()) {
        out.unsafeGetTensorImpl()->set_data(
            const_cast<Tensor&>(cache).mutable_data_ptr());
    }
    return out;
}

Tensor& update_mhsa_kv_out(
    RuntimeContext& ctx,
    const Tensor& cache,
    const Tensor& k_val,
    const Tensor& v_val,
    int64_t start_pos,
    int64_t layer_idx,
    Tensor& out)
{
    int64_t B = cache.size(2);
    int64_t maxlen = cache.size(3);

    int64_t head_elements = 1;
    for (size_t i = 4; i < cache.dim(); ++i) {
        head_elements *= cache.size(i);
    }
    size_t element_size = cache.element_size();

    int64_t per_layer_stride = 2 * B * maxlen * head_elements;
    int64_t per_channel_stride = B * maxlen * head_elements;
    char* dst_ptr = const_cast<Tensor&>(cache).mutable_data_ptr<char>();

    char* layer_base = dst_ptr + layer_idx * per_layer_stride * element_size;

    copy_kv_slice(
        k_val.const_data_ptr<char>(),
        layer_base,
        k_val.size(0), k_val.size(1), maxlen, start_pos,
        head_elements, element_size);

    copy_kv_slice(
        v_val.const_data_ptr<char>(),
        layer_base + per_channel_stride * element_size,
        v_val.size(0), v_val.size(1), maxlen, start_pos,
        head_elements, element_size);

    if (out.const_data_ptr() != cache.const_data_ptr()) {
        out.unsafeGetTensorImpl()->set_data(
            const_cast<Tensor&>(cache).mutable_data_ptr());
    }
    return out;
}

} // namespace phono::ops

EXECUTORCH_LIBRARY(phono, "update_kv_cache.out", phono::ops::update_kv_cache_out);
EXECUTORCH_LIBRARY(phono, "update_cross_kv.out", phono::ops::update_cross_kv_out);
EXECUTORCH_LIBRARY(phono, "update_mhsa_kv.out", phono::ops::update_mhsa_kv_out);