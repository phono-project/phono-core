// tokenizer.hpp
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "algo/trie.hpp"

namespace phono::core {

class ModelPackageConfig;  // fwd decl (config.hpp)

class Tokenizer {
public:
    // Builds directly from three vocab file paths + ordered special token
    // names (appended at the end of context_vocab).
    Tokenizer(const std::string& chinese_vocab_path, const std::string& context_vocab_path,
              const std::string& pinyin_vocab_path, const std::vector<std::string>& special_tokens_def);

    // Convenience constructor: resolves vocab paths from a loaded
    // ModelPackageConfig.
    static Tokenizer from_config(const ModelPackageConfig& cfg);

    int32_t chinese_vocab_size() const { return static_cast<int32_t>(chinese_vocab_.size()); }
    int32_t pinyin_vocab_size() const { return static_cast<int32_t>(pinyin_vocab_.size()); }
    int32_t context_vocab_size() const { return context_size_; }

    // Returns the global context-vocab id for a named special token
    // (e.g. "bos_token").
    // Throws std::out_of_range if the name wasn't declared in config.
    int32_t special_token_id(const std::string& name) const;

    // Encode UTF-8 `text` one codepoint at a time via context_vocab.
    // Unknown codepoints are silently skipped.
    std::vector<int32_t> encode_context(const std::string& text_utf8) const;

    // Encode a list of pinyin syllables via pinyin_vocab; syllables not in
    // vocab fall back to the nearest match by edit distance.
    std::vector<int32_t> encode_pinyin(const std::vector<std::string>& pinyin_list) const;

    // Greedily split unseparated pinyin using the longest vocabulary match.
    // A single quote forces a syllable boundary and is not included in output.
    std::vector<std::string> separate_greedy(const std::string& pinyin) const;

    // Convert a sequence of chinese_vocab ids back into UTF-8 text.
    // Unknown ids are silently skipped.
    std::string ids_to_text(const std::vector<int32_t>& ids) const;

    // Single-id -> UTF-8 char lookup used when building Viterbi candidate
    // maps from per-position logits. Returns "" if id is out of range.
    std::string id_to_chinese(int32_t id) const;

    // Maps a Chinese-vocabulary id back to the corresponding context-vocab
    // id for feeding generated characters into the decoder.
    int32_t chinese_id_to_context_id(int32_t id) const;

private:
    static int edit_distance(const std::string& a, const std::string& b);

    std::unordered_map<std::string, int32_t> chinese_vocab_;
    std::unordered_map<int32_t, std::string> id_to_chinese_;

    std::unordered_map<std::string, int32_t> pinyin_vocab_;
    std::vector<std::string> pinyin_list_;  // id -> syllable, for edit-distance fallback
    algo::Trie pinyin_tree_;

    std::unordered_map<std::string, int32_t> context_vocab_;  // includes special tokens
    int32_t context_base_size_ = 0;                            // size before special tokens
    int32_t context_size_ = 0;                                 // size including special tokens

    std::unordered_map<std::string, int32_t> special_tokens_;  // name -> global context id
};

}  // namespace phono::core
