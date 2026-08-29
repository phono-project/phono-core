// utf8_util.hpp
#pragma once

#include <string>
#include <vector>

#include <uni_algo/conv.h>

namespace phono::core {

inline std::vector<std::string> utf8_split_chars(const std::string& text) {
    const std::u32string cps = una::utf8to32<char, char32_t>(text);
    std::vector<std::string> out;
    out.reserve(cps.size());
    for (char32_t cp : cps) {
        out.emplace_back(una::utf32to8(std::u32string(1, cp)));
    }
    return out;
}

}  // namespace phono::core
