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

size_t Trie::longest_match(std::string_view text, size_t offset) const {
    if (offset >= text.size()) {
        return 0;
    }

    const Node* node = &root_;
    size_t longest = 0;
    for (size_t i = offset; i < text.size(); ++i) {
        const auto it = node->children.find(text[i]);
        if (it == node->children.end()) {
            break;
        }
        node = it->second.get();
        if (node->terminal) {
            longest = i - offset + 1;
        }
    }
    return longest;
}

}  // namespace phono::algo
