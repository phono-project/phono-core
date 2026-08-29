#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace phono::algo {

class Trie {
public:
    void insert(std::string_view word);
    bool contains(std::string_view word) const;

    // Returns the length of the longest word starting at offset, or zero.
    size_t longest_match(std::string_view text, size_t offset = 0) const;

private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> children;
        bool terminal = false;
    };

    Node root_;
};

}  // namespace phono::algo
