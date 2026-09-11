#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "algo/trie.hpp"

namespace phono::algo {

struct PinyinEdge {
    size_t begin = 0;
    size_t end = 0;
    bool invalid = false;
};

struct SegmentationPath {
    bool reachable = false;
    int32_t invalid_char_count = 0;
    double gap_score = 0.0;
    std::vector<PinyinEdge> edges;
};

// FMM compatibility path with explicit invalid edges instead of silent
// single-character acceptance.
SegmentationPath separate_fmm_checked(const std::string& input, const Trie& trie,
                                      const std::vector<bool>& forced_boundaries);

// Exact DAG Viterbi. When allow_invalid is true, single-character invalid
// edges guarantee reachability. Paths are ordered lexicographically by fewest
// invalid characters, highest gap score, fewest tokens, then longest earliest
// token.
SegmentationPath decode_gap_viterbi(const std::string& input, const Trie& trie,
                                    const std::vector<bool>& forced_boundaries,
                                    const std::vector<float>& gap_logits,
                                    bool allow_invalid);

}  // namespace phono::algo
