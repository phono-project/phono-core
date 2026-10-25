#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <cstring>
#include <tuple>

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

std::tuple<Tensor&, Tensor&> read_mhsa_kv_out(
    RuntimeContext& ctx, const Tensor& cache, int64_t end_pos,
    int64_t layer_idx, const Tensor& positions, Tensor& out0, Tensor& out1) {
    auto result = std::tie(out0, out1);
    ET_KERNEL_CHECK(ctx, cache.dim() == 6 && cache.size(1) == 2,
                    InvalidArgument, result);
    ET_KERNEL_CHECK(ctx, layer_idx >= 0 && layer_idx < cache.size(0) &&
                    positions.dim() == 1 && positions.numel() <= cache.size(3) &&
                    end_pos >= 0 && end_pos <= positions.numel(), InvalidArgument, result);
    ET_KERNEL_CHECK(ctx, torch::executor::tensor_is_default_dim_order(cache) &&
                    torch::executor::tensor_is_default_dim_order(out0) &&
                    torch::executor::tensor_is_default_dim_order(out1) &&
                    cache.scalar_type() == out0.scalar_type() &&
                    cache.scalar_type() == out1.scalar_type(), InvalidArgument, result);
    Tensor::SizesType sizes[] = {static_cast<Tensor::SizesType>(cache.size(2)),
                                 static_cast<Tensor::SizesType>(positions.numel()),
                                 static_cast<Tensor::SizesType>(cache.size(4)),
                                 static_cast<Tensor::SizesType>(cache.size(5))};
    ET_KERNEL_CHECK(ctx, executorch::runtime::resize_tensor(out0, {sizes, 4}) == executorch::runtime::Error::Ok &&
                    executorch::runtime::resize_tensor(out1, {sizes, 4}) == executorch::runtime::Error::Ok,
                    InvalidArgument, result);
    const size_t token_bytes = cache.size(4) * cache.size(5) * cache.element_size();
    const size_t valid_bytes = static_cast<size_t>(end_pos) * token_bytes;
    const size_t batch_bytes = positions.numel() * token_bytes;
    const auto strides = cache.strides();
    const auto* source = cache.const_data_ptr<char>();
    Tensor* outputs[] = {&out0, &out1};
    for (int64_t kv = 0; kv < 2; ++kv) {
        auto* destination = outputs[kv]->mutable_data_ptr<char>();
        for (int64_t batch = 0; batch < cache.size(2); ++batch) {
            const size_t offset = (layer_idx * strides[0] + kv * strides[1] +
                                   batch * strides[2]) * cache.element_size();
            if (valid_bytes) std::memcpy(destination, source + offset, valid_bytes);
            if (batch_bytes > valid_bytes)
                std::memset(destination + valid_bytes, 0, batch_bytes - valid_bytes);
            destination += batch_bytes;
        }
    }
    return result;
}

} // namespace phono::ops

EXECUTORCH_LIBRARY(phono, "update_mhsa_kv.out", phono::ops::update_mhsa_kv_out);

namespace {
// ExecuTorch's convenience EXECUTORCH_LIBRARY wrapper supports one mutable
// output. Use the ordinary boxed registration API for the K/V pair.
void read_mhsa_kv_boxed(executorch::runtime::KernelRuntimeContext& ctx,
                       executorch::runtime::Span<executorch::runtime::EValue*> args) {
    phono::ops::read_mhsa_kv_out(ctx, args[0]->toTensor(), args[1]->toInt(),
                                args[2]->toInt(), args[3]->toTensor(), args[4]->toTensor(), args[5]->toTensor());
    executorch::runtime::internal::event_tracer_log_evalue(ctx.internal_event_tracer(), *args[4]);
    executorch::runtime::internal::event_tracer_log_evalue(ctx.internal_event_tracer(), *args[5]);
}
const auto register_read_mhsa_kv = executorch::runtime::register_kernel(
    executorch::runtime::Kernel("phono::read_mhsa_kv.out", read_mhsa_kv_boxed));
}
