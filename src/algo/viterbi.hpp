// viterbi.hpp
//
// C++ implementation of viterbi_nbest.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "algo/trie.hpp"  // WordMatch

namespace phono::algo {

struct ViterbiResult {
    double score = 0.0;
    std::vector<std::string> words;  // ordered list of single chars / dictionary words
};

// candidates[i]: {char -> prob} at position i (post-softmax, epsilon-filtered).
// words_at[i]:   dictionary words that can start at position i (from find_matching_words_xxx).
// Returns up to N results sorted by score descending (fewer if the beam
// collapses to fewer distinct full-length paths).
std::vector<ViterbiResult> viterbi_nbest(const std::vector<std::unordered_map<std::string, double>>& candidates,
                                          const std::vector<std::vector<WordMatch>>& words_at, double beta_single,
                                          double beta_word, int N);

}  // namespace phono::algo
