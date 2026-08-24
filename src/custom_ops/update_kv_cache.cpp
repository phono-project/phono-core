#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <cstring>

namespace phono::ops {

using torch::executor::Tensor;
using torch::executor::RuntimeContext;

Tensor& update_mhsa_kv_out(
    RuntimeContext& ctx,
    const Tensor& cache,
    const Tensor& k_val,
    const Tensor& v_val,
    int64_t start_pos,
    int64_t layer_idx,
    Tensor& out)
{
    int64_t head_elements = 1;
    for (size_t i = 4; i < cache.dim(); ++i) {
        head_elements *= cache.size(i);
    }
    size_t element_size = cache.element_size();

    const auto cache_strides = cache.strides();
    const auto k_strides = k_val.strides();
    const auto v_strides = v_val.strides();
    char* dst_ptr = const_cast<Tensor&>(cache).mutable_data_ptr<char>();
    const char* k_ptr = k_val.const_data_ptr<char>();
    const char* v_ptr = v_val.const_data_ptr<char>();
    const size_t copy_bytes = static_cast<size_t>(head_elements) * element_size;

    // Use tensor strides instead of assuming a contiguous cache so this op
    // remains correct for cache views as well as the normal B-wide tensor.
    for (int64_t b = 0; b < k_val.size(0); ++b) {
        for (int64_t s = 0; s < k_val.size(1); ++s) {
            const int64_t cache_offset =
                layer_idx * cache_strides[0] + b * cache_strides[2] +
                (start_pos + s) * cache_strides[3];
            const int64_t k_offset = b * k_strides[0] + s * k_strides[1];
            const int64_t v_offset = b * v_strides[0] + s * v_strides[1];
            std::memcpy(dst_ptr + cache_offset * element_size,
                        k_ptr + k_offset * element_size, copy_bytes);
            std::memcpy(dst_ptr + (cache_offset + cache_strides[1]) * element_size,
                        v_ptr + v_offset * element_size, copy_bytes);
        }
    }

    if (out.const_data_ptr() != cache.const_data_ptr()) {
        out.unsafeGetTensorImpl()->set_data(
            const_cast<Tensor&>(cache).mutable_data_ptr());
    }
    return out;
}

} // namespace phono::ops

EXECUTORCH_LIBRARY(phono, "update_mhsa_kv.out", phono::ops::update_mhsa_kv_out);
