#include "engine/pinyin_segmenter.hpp"

#include <fstream>
#include <stdexcept>

#include <executorch/extension/tensor/tensor.h>

namespace phono::engine {

using executorch::aten::ScalarType;
using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::runtime::EValue;
using executorch::runtime::Tag;

PinyinSegmentScorer::PinyinSegmentScorer(const core::ModelPackageConfig& package,
                                         const core::SegmenterConfig& config)
    : config_(config), module_(std::make_unique<Module>(package.resolve(config.model_path))) {
    std::ifstream vocab(package.resolve(config.char_vocab));
    if (!vocab.is_open()) {
        throw std::runtime_error("PinyinSegmentScorer: cannot open character vocabulary");
    }
    std::string token;
    int64_t id = 0;
    while (std::getline(vocab, token)) {
        if (!token.empty() && token.back() == '\r') token.pop_back();
        if (token == "<unk>") unknown_id_ = id;
        if (token.size() == 1) {
            char_ids_.emplace(static_cast<unsigned char>(token.front()), id);
        }
        ++id;
    }
    if (unknown_id_ < 0) {
        throw std::runtime_error("PinyinSegmentScorer: character vocabulary has no <unk>");
    }

    auto names = module_->method_names();
    if (!names.ok() || names->count(config_.method) == 0) {
        throw std::runtime_error("PinyinSegmentScorer: configured method is missing");
    }
    auto meta = module_->method_meta(config_.method);
    if (!meta.ok() || meta->num_inputs() != 1 || meta->num_outputs() != 1 ||
        !meta->input_tag(0).ok() || meta->input_tag(0).get() != Tag::Tensor ||
        !meta->output_tag(0).ok() || meta->output_tag(0).get() != Tag::Tensor) {
        throw std::runtime_error("PinyinSegmentScorer: expected Tensor -> Tensor method");
    }
    const auto error = module_->load_method(config_.method);
    if (error != executorch::runtime::Error::Ok) {
        throw std::runtime_error(
            "PinyinSegmentScorer: failed to load configured method (error " +
            std::to_string(static_cast<int>(error)) + ")");
    }
}

PinyinSegmentScorer::~PinyinSegmentScorer() = default;

std::vector<float> PinyinSegmentScorer::score(const std::string& canonical_input) const {
    const int32_t length = static_cast<int32_t>(canonical_input.size());
    if (length < config_.min_input_chars || length > config_.max_input_chars) {
        throw std::invalid_argument("PinyinSegmentScorer: input length is outside model limits");
    }
    std::vector<int64_t> ids;
    ids.reserve(canonical_input.size());
    for (const unsigned char value : canonical_input) {
        const auto found = char_ids_.find(value);
        ids.push_back(found == char_ids_.end() ? unknown_id_ : found->second);
    }
    auto input = make_tensor_ptr(std::vector<int32_t>{1, length}, ids.data(), ScalarType::Long);
    auto result = module_->execute(config_.method, {EValue(*input)});
    if (!result.ok() || result->size() != 1 || !result->at(0).isTensor()) {
        throw std::runtime_error("PinyinSegmentScorer: model execution failed");
    }
    const auto& output = result->at(0).toTensor();
    const auto sizes = output.sizes();
    const bool valid_shape =
        (sizes.size() == 2 && sizes[0] == 1 && sizes[1] == length - 1) ||
        (sizes.size() == 1 && sizes[0] == length - 1);
    if (!valid_shape || output.scalar_type() != ScalarType::Float) {
        throw std::runtime_error("PinyinSegmentScorer: expected float logits shaped [1, L-1]");
    }
    const float* data = output.const_data_ptr<float>();
    return {data, data + length - 1};
}

}  // namespace phono::engine
