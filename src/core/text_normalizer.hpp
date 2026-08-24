#pragma once

#include <string>

namespace phono::core {

// Applies the same Chinese simplification and Unicode NFKC normalization used
// by the training data pipeline.
std::string normalize_text_utf8(const std::string& text_utf8);

}  // namespace phono::core
