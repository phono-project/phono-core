#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/module/module.h>

#include "core/config.hpp"

namespace phono::engine {

// Thin owner for the optional ExecuTorch gap scorer. The model consumes one
// character-id sequence [1, L] and returns L-1 boundary logits. Legality and
// path selection deliberately remain in src/algo rather than in this class.
class PinyinSegmentScorer {
public:
    PinyinSegmentScorer(const core::ModelPackageConfig& package,
                        const core::SegmenterConfig& config);
    ~PinyinSegmentScorer();

    PinyinSegmentScorer(const PinyinSegmentScorer&) = delete;
    PinyinSegmentScorer& operator=(const PinyinSegmentScorer&) = delete;

    std::vector<float> score(const std::string& canonical_input) const;
    int32_t min_input_chars() const { return config_.min_input_chars; }
    int32_t max_input_chars() const { return config_.max_input_chars; }

private:
    core::SegmenterConfig config_;
    std::unordered_map<unsigned char, int64_t> char_ids_;
    int64_t unknown_id_ = -1;
    std::unique_ptr<executorch::extension::Module> module_;
};

}  // namespace phono::engine
