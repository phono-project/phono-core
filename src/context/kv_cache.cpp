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

}  // namespace phono::context
