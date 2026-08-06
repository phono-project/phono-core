#include "trie.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace phono::algo {

nlohmann::json load_trie(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("load_trie: cannot open " + path);
    }
    nlohmann::json trie;
    in >> trie;
    return trie;
}

namespace {

struct StackEntry {
    const nlohmann::json* node;
    int pos;
    int depth;
    std::string chars;
    double log_prob;
};

}  // namespace

std::vector<WordMatch> find_matching_words(const nlohmann::json& trie,
                                            const std::vector<std::unordered_map<std::string, double>>& candidates,
                                            int start) {
    std::vector<WordMatch> results;
    std::vector<StackEntry> stack;
    stack.reserve(256);
    stack.push_back(StackEntry{&trie, start, 0, std::string(), 0.0});

    while (!stack.empty()) {
        StackEntry entry = std::move(stack.back());
        stack.pop_back();

        if (entry.pos < 0 || static_cast<size_t>(entry.pos) >= candidates.size()) {
            continue;
        }
        if (!entry.node->is_object()) {
            continue;
        }

        const auto& cand_dict = candidates[static_cast<size_t>(entry.pos)];

        for (const auto& [ch, prob] : cand_dict) {
            if (prob <= 0.0) continue;
            if (!entry.node->contains(ch)) continue;

            std::string new_chars = entry.chars + ch;
            int new_depth = entry.depth + 1;
            double new_log_prob = entry.log_prob + std::log(prob);

            const nlohmann::json& child = (*entry.node)[ch];
            if (child.is_object() && child.contains("#")) {
                results.push_back(WordMatch{new_depth, new_chars, new_log_prob});
            }
            stack.push_back(StackEntry{&child, entry.pos + 1, new_depth, new_chars, new_log_prob});
        }
    }

    return results;
}

}  // namespace phono::algo
