#include "text_normalizer.hpp"

#include <string>

#include <uni_algo/norm.h>

#include "algo/zh2hans.hpp"

namespace phono::core {

std::string normalize_text_utf8(const std::string& text_utf8) {
    if (text_utf8.empty()) {
        return {};
    }

    // Same order as the training data pipeline: Traditional -> Simplified
    // first, then NFKC.
    const std::string simplified = algo::to_simplified_zh(text_utf8);
    return una::norm::to_nfkc_utf8(simplified);
}

}  // namespace phono::core
