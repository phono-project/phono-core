#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace phono::core {

struct PinyinNormalizationConfig {
    bool lowercase_ascii = true;
    bool normalize_v_to_u = true;
    std::string separators = "'";
};

struct PinyinNormalizationEvent {
    std::string type;
    size_t begin = 0;  // half-open byte range in the original input
    size_t end = 0;
    std::string before;
    std::string after;
};

struct NormalizedPinyin {
    std::string canonical_input;
    // original byte offset for each byte in canonical_input
    std::vector<size_t> source_offsets;
    // boundary k lies between canonical_input[k] and canonical_input[k + 1]
    std::vector<bool> forced_boundaries;
    std::vector<PinyinNormalizationEvent> events;
};

NormalizedPinyin normalize_pinyin(const std::string& input,
                                  const PinyinNormalizationConfig& config);

}  // namespace phono::core
