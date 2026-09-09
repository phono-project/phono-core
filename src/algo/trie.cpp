#include "trie.hpp"

namespace phono::algo {

void Trie::insert(std::string_view word) {
    Node* node = &root_;
    for (const char ch : word) {
        auto& child = node->children[ch];
        if (!child) {
            child = std::make_unique<Node>();
        }
        node = child.get();
    }
    if (!word.empty()) {
        node->terminal = true;
    }
}

bool Trie::contains(std::string_view word) const {
    const Node* node = &root_;
    for (const char ch : word) {
        const auto it = node->children.find(ch);
        if (it == node->children.end()) {
            return false;
        }
        node = it->second.get();
    }
    return !word.empty() && node->terminal;
}

std::vector<size_t> Trie::match_lengths(std::string_view text, size_t offset,
                                        size_t end) const {
    std::vector<size_t> matches;
    if (offset >= text.size()) {
        return matches;
    }
    end = std::min(end, text.size());

    const Node* node = &root_;
    for (size_t i = offset; i < end; ++i) {
        const auto it = node->children.find(text[i]);
        if (it == node->children.end()) {
            break;
        }
        node = it->second.get();
        if (node->terminal) {
            matches.push_back(i - offset + 1);
        }
    }
    return matches;
}

size_t Trie::longest_match(std::string_view text, size_t offset) const {
    const auto matches = match_lengths(text, offset);
    return matches.empty() ? 0 : matches.back();
}

}  // namespace phono::algo
