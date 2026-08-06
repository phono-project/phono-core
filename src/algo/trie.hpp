// trie.hpp
#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace phono::algo {

// One dictionary word match found starting at some position: its length
// (in characters), the word text (UTF-8), and the summed log-prob of its
// characters under the model's per-position distributions.
struct WordMatch {
    int length = 0;
    std::string word;
    double log_prob = 0.0;
};

// Loads a pre-built trie from a JSON file.
nlohmann::json load_trie(const std::string& path);

// Finds all dictionary words that can start at position `start`, given
// per-position character candidate probabilities.
// `candidates[i]` maps a UTF-8 character to its probability at position i.
std::vector<WordMatch> find_matching_words(const nlohmann::json& trie,
                                            const std::vector<std::unordered_map<std::string, double>>& candidates,
                                            int start);

}  // namespace phono::algo
