// config.hpp
//
// Loads the on-device model package's `config.json`.
//
// Expected package layout on disk:
//
//   <package_root>/
//     config.json
//     bins/
//       pre_model.pte
//       post_model.pte
//     vocabs/
//       chinese_vocab.txt
//       context_vocab.txt
//       pinyin_vocab.txt
//     dict/
//       dict_trie.json
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace phono::core {

// Mirrors base.yaml's `common:` section.
struct CommonConfig {
    int32_t model_dim = 768;
    double rope_theta = 1000.0;
};

// Mirrors base.yaml's `pre_model:` section.
struct PreModelDims {
    int32_t max_seqlen = 128;
    int32_t mhsa_layers = 8;
    int32_t mhsa_heads = 4;
    int32_t attn_dim = 256;
    int32_t ffn_common_dim = 4096;

    int32_t self_head_dim() const { return attn_dim / mhsa_heads; }
};

// Mirrors base.yaml's `post_model:` section.
struct PostModelDims {
    int32_t max_seqlen = 32;
    int32_t mhsa_layers = 12;
    int32_t mhsa_heads = 4;
    int32_t attn_dim = 256;
    int32_t mhca_heads = 12;
    int32_t mhca_attn_dim = 768;
    int32_t ffn_common_dim = 4096;  // unused on-device, kept for completeness

    int32_t self_head_dim() const { return attn_dim / mhsa_heads; }
    int32_t cross_head_dim() const { return mhca_attn_dim / mhca_heads; }
};

// Mirrors the tokenizer's own YAML `vocabs:` section.
struct VocabPaths {
    std::string chinese_vocab = "vocabs/chinese_vocab.txt";
    std::string context_vocab = "vocabs/context_vocab.txt";
    std::string pinyin_vocab = "vocabs/pinyin_vocab.txt";
    std::vector<std::string> context_special_tokens = {"bos_token"};
};

// Viterbi decoding hyperparameters.
struct DecodingParams {
    double beta_single = 0.4636;
    double beta_word = 0.4839;
    double epsilon = 0.001;
    int n_best = 3;
    std::string trie_path = "dict/dict_trie.json";
};

// Runtime / packaging knobs.
struct RuntimeParams {
    int32_t batch_size = 1;                // B in the exported cache tensors.
    std::string cache_dtype = "float32";   // dtype of KV-cache buffers;
                                           // the .pte graphs were exported/lowered with.
    std::string pre_model_path = "bins/pre_model.pte";
    std::string post_model_path = "bins/post_model.pte";
    std::string pre_pass1_method = "pre_model_pass1";
    std::string pre_pass2_method = "pre_model_pass2";
    std::string post_method = "post_model";
};

// Top level configuration struct.
class ModelPackageConfig {
public:
    // Loads and validates `<package_root>/config.json`.
    static ModelPackageConfig load(const std::string& package_root);

    // Resolves a path from config.json (e.g. "bins/pre_model.pte") against
    // package_root. Absolute input paths are returned unchanged.
    std::string resolve(const std::string& relative_path) const;

    CommonConfig common;
    PreModelDims pre_model;
    PostModelDims post_model;
    VocabPaths vocabs;
    DecodingParams decoding;
    RuntimeParams runtime;

    std::string package_root;
};

}  // namespace phono::core
