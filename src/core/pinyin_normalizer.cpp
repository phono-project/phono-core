#include "pinyin_normalizer.hpp"

#include <algorithm>
#include <cctype>

namespace phono::core {

NormalizedPinyin normalize_pinyin(const std::string& input,
                                  const PinyinNormalizationConfig& config) {
    NormalizedPinyin result;
    bool force_before_next = false;

    for (size_t source = 0; source < input.size(); ++source) {
        const unsigned char raw = static_cast<unsigned char>(input[source]);
        if (config.separators.find(static_cast<char>(raw)) != std::string::npos) {
            if (!result.canonical_input.empty()) {
                force_before_next = true;
            }
            result.events.push_back(PinyinNormalizationEvent{
                "forced_boundary", source, source + 1,
                std::string(1, static_cast<char>(raw)), ""});
            continue;
        }

        char normalized = static_cast<char>(raw);
        if (config.lowercase_ascii && raw >= 'A' && raw <= 'Z') {
            normalized = static_cast<char>(std::tolower(raw));
            result.events.push_back(PinyinNormalizationEvent{
                "ascii_lowercase", source, source + 1,
                std::string(1, static_cast<char>(raw)), std::string(1, normalized)});
        }

        if (!result.canonical_input.empty()) {
            result.forced_boundaries.push_back(force_before_next);
        }
        const bool separated = force_before_next;
        force_before_next = false;

        if (config.normalize_v_to_u && normalized == 'v' && !separated &&
            !result.canonical_input.empty()) {
            const char previous = result.canonical_input.back();
            if (previous == 'j' || previous == 'q' || previous == 'x' || previous == 'y') {
                normalized = 'u';
                result.events.push_back(PinyinNormalizationEvent{
                    "v_to_u", source, source + 1, "v", "u"});
            }
        }

        result.canonical_input.push_back(normalized);
        result.source_offsets.push_back(source);
    }
    return result;
}

}  // namespace phono::core
