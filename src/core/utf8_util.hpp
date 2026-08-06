// utf8_util.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace phono::core {

inline std::vector<std::string> utf8_split_chars(const std::string& text) {
    std::vector<std::string> out;
    out.reserve(text.size());

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if ((c & 0x80u) == 0x00u) {
            len = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            len = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            len = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            len = 4;
        } else {
            len = 1;  // stray continuation/invalid byte: emit as-is
        }
        if (i + len > n) {
            len = n - i;  // truncated sequence at end of buffer
        }
        out.emplace_back(text.substr(i, len));
        i += len;
    }
    return out;
}

}  // namespace phono::core
