#include "pinyin_normalizer.hpp"

#include <algorithm>
#include <cctype>
#include <iterator>

#include <uni_algo/conv.h>
#include <uni_algo/ranges_conv.h>

namespace phono::core {

NormalizedPinyin normalize_pinyin(const std::string& input,
                                  const PinyinNormalizationConfig& config) {
    NormalizedPinyin result;
    bool force_before_next = false;
    const std::u32string separators = una::utf8to32<char, char32_t>(config.separators);
    auto codepoints = input | una::views::utf8;

    for (auto it = codepoints.begin(); it != codepoints.end(); ++it) {
        const size_t source_begin = static_cast<size_t>(std::distance(input.begin(), it.begin()));
        const size_t source_end = static_cast<size_t>(std::distance(input.begin(), it.end()));
        const char32_t codepoint = *it;
        const std::string original = input.substr(source_begin, source_end - source_begin);
        if (std::find(separators.begin(), separators.end(), codepoint) != separators.end()) {
            if (!result.canonical_input.empty()) {
                force_before_next = true;
            }
            result.events.push_back(PinyinNormalizationEvent{
                "forced_boundary", source_begin, source_end, original, ""});
            continue;
        }

        std::string normalized = original;
        if (config.lowercase_ascii && codepoint >= U'A' && codepoint <= U'Z') {
            normalized.assign(1, static_cast<char>(std::tolower(static_cast<unsigned char>(codepoint))));
            result.events.push_back(PinyinNormalizationEvent{
                "ascii_lowercase", source_begin, source_end, original, normalized});
        }

        const bool separated = force_before_next;
        if (config.normalize_v_to_u && normalized == "v" && !separated &&
            !result.canonical_input.empty()) {
            const char previous = result.canonical_input.back();
            if (previous == 'j' || previous == 'q' || previous == 'x' || previous == 'y') {
                normalized = "u";
                result.events.push_back(PinyinNormalizationEvent{
                    "v_to_u", source_begin, source_end, "v", "u"});
            }
        }

        for (size_t byte = 0; byte < normalized.size(); ++byte) {
            if (!result.canonical_input.empty()) {
                result.forced_boundaries.push_back(byte == 0 && force_before_next);
            }
            result.canonical_input.push_back(normalized[byte]);
            result.source_offsets.push_back(source_begin + std::min(byte, original.size() - 1));
        }
        force_before_next = false;
    }
    return result;
}

}  // namespace phono::core
