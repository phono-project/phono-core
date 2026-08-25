#include "algo/zh2hans.hpp"

#include <algorithm>
#include <cstdint>

#include <uni_algo/conv.h>

#include "gen/zh2hansdict.h"

namespace phono {
namespace algo {

namespace {

const zh2hans::FirstCpEntry* find_first_cp_group(std::uint32_t cp) {
    const auto it = std::lower_bound(
        zh2hans::kFirstCpIndex.begin(), zh2hans::kFirstCpIndex.end(), cp,
        [](const zh2hans::FirstCpEntry& entry, std::uint32_t value) {
            return entry.cp < value;
        });
    if (it != zh2hans::kFirstCpIndex.end() && it->cp == cp) {
        return &*it;
    }
    return nullptr;
}

}  // namespace

std::string to_simplified_zh(const std::string& text_utf8) {
    if (text_utf8.empty()) {
        return {};
    }

    // Work in code-point space; uni-algo maps ill-formed sequences to U+FFFD
    // following the W3C recommendations, so arbitrary input stays safe.
    const std::u32string src = una::utf8to32<char, char32_t>(text_utf8);

    std::string out;
    out.reserve(text_utf8.size());

    std::u32string pending;
    const std::size_t n = src.size();
    std::size_t i = 0;
    while (i < n) {
        const zh2hans::Rule* matched = nullptr;
        if (const zh2hans::FirstCpEntry* group = find_first_cp_group(src[i])) {
            for (std::size_t r = group->begin; r < group->end; ++r) {
                const zh2hans::Rule& rule = zh2hans::kRules[r];
                // Each group is sorted by key length descending, so the first
                // prefix match is the longest (maximal-munch).
                if (i + rule.from.size() <= n &&
                    src.compare(i, rule.from.size(), rule.from) == 0) {
                    matched = &rule;
                    break;
                }
            }
        }

        if (matched != nullptr) {
            if (!pending.empty()) {
                out += una::utf32to8(pending);
                pending.clear();
            }
            out.append(matched->to);
            i += matched->from.size();
        } else {
            pending.push_back(src[i]);
            ++i;
        }
    }
    if (!pending.empty()) {
        out += una::utf32to8(pending);
    }
    return out;
}

}  // namespace algo
}  // namespace phono
