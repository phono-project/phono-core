#pragma once

#include <string>

namespace phono {
namespace algo {

// Converts Traditional Chinese text to Simplified Chinese by applying the
// generated zh2Hans replacement table (src/gen/zh2hansdict.h) with a
// longest-match (maximal-munch) scan. Ill-formed UTF-8 input is handled
// safely: unrecognized sequences are mapped to U+FFFD and left untouched by
// the dictionary.
std::string to_simplified_zh(const std::string& text_utf8);

}  // namespace algo
}  // namespace phono
