#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace phono::algo {

class Trie {
public:
    void insert(std::string_view word);
    bool contains(std::string_view word) const;

    // Returns every terminal match starting at offset, shortest first. No
    // match may extend beyond end (defaults to text.size()).
    std::vector<size_t> match_lengths(std::string_view text, size_t offset = 0,
                                      size_t end = std::string_view::npos) const;

private:
    struct Node {
        std::unordered_map<char, std::unique_ptr<Node>> children;
        bool terminal = false;
    };

    Node root_;
};

}  // namespace phono::algo
