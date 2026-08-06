#include "config.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using nlohmann::json;

namespace phono::core {

namespace {

template <typename T>
T get_or(const json& j, const char* key, T default_value) {
    if (j.is_object() && j.contains(key) && !j.at(key).is_null()) {
        return j.at(key).get<T>();
    }
    return default_value;
}

}  // namespace

std::string ModelPackageConfig::resolve(const std::string& relative_path) const {
    fs::path p(relative_path);
    if (p.is_absolute()) {
        return p.string();
    }
    return (fs::path(package_root) / p).lexically_normal().string();
}

ModelPackageConfig ModelPackageConfig::load(const std::string& package_root) {
    ModelPackageConfig cfg;
    cfg.package_root = package_root;

    const fs::path config_path = fs::path(package_root) / "config.json";
    std::ifstream in(config_path);
    if (!in.is_open()) {
        throw std::runtime_error("ModelPackageConfig::load: cannot open " + config_path.string());
    }

    json root;
    try {
        in >> root;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "ModelPackageConfig::load: failed to parse " << config_path << ": " << e.what();
        throw std::runtime_error(oss.str());
    }

    // common
    if (root.contains("common")) {
        const json& c = root.at("common");
        cfg.common.model_dim = get_or<int32_t>(c, "model_dim", cfg.common.model_dim);
        cfg.common.rope_theta = get_or<double>(c, "rope_theta", cfg.common.rope_theta);
    }

    // pre_model
    if (root.contains("pre_model")) {
        const json& p = root.at("pre_model");
        cfg.pre_model.max_seqlen = get_or<int32_t>(p, "max_seqlen", cfg.pre_model.max_seqlen);
        cfg.pre_model.mhsa_layers = get_or<int32_t>(p, "mhsa_layers", cfg.pre_model.mhsa_layers);
        cfg.pre_model.mhsa_heads = get_or<int32_t>(p, "mhsa_heads", cfg.pre_model.mhsa_heads);
        cfg.pre_model.attn_dim = get_or<int32_t>(p, "attn_dim", cfg.pre_model.attn_dim);
        cfg.pre_model.ffn_common_dim = get_or<int32_t>(p, "ffn_common_dim", cfg.pre_model.ffn_common_dim);
    }

    // post_model
    if (root.contains("post_model")) {
        const json& p = root.at("post_model");
        cfg.post_model.max_seqlen = get_or<int32_t>(p, "max_seqlen", cfg.post_model.max_seqlen);
        cfg.post_model.mhsa_layers = get_or<int32_t>(p, "mhsa_layers", cfg.post_model.mhsa_layers);
        cfg.post_model.mhsa_heads = get_or<int32_t>(p, "mhsa_heads", cfg.post_model.mhsa_heads);
        cfg.post_model.attn_dim = get_or<int32_t>(p, "attn_dim", cfg.post_model.attn_dim);
        cfg.post_model.mhca_heads = get_or<int32_t>(p, "mhca_heads", cfg.post_model.mhca_heads);
        cfg.post_model.mhca_attn_dim = get_or<int32_t>(p, "mhca_attn_dim", cfg.post_model.mhca_attn_dim);
        cfg.post_model.ffn_common_dim = get_or<int32_t>(p, "ffn_common_dim", cfg.post_model.ffn_common_dim);
    }

    // vocabs
    if (root.contains("vocabs")) {
        const json& v = root.at("vocabs");
        cfg.vocabs.chinese_vocab = get_or<std::string>(v, "chinese_vocab", cfg.vocabs.chinese_vocab);
        cfg.vocabs.context_vocab = get_or<std::string>(v, "context_vocab", cfg.vocabs.context_vocab);
        cfg.vocabs.pinyin_vocab = get_or<std::string>(v, "pinyin_vocab", cfg.vocabs.pinyin_vocab);
        if (v.contains("context_special_tokens") && v.at("context_special_tokens").is_array()) {
            cfg.vocabs.context_special_tokens =
                v.at("context_special_tokens").get<std::vector<std::string>>();
        }
    }

    // decoding
    if (root.contains("decoding")) {
        const json& d = root.at("decoding");
        cfg.decoding.beta_single = get_or<double>(d, "beta_single", cfg.decoding.beta_single);
        cfg.decoding.beta_word = get_or<double>(d, "beta_word", cfg.decoding.beta_word);
        cfg.decoding.epsilon = get_or<double>(d, "epsilon", cfg.decoding.epsilon);
        cfg.decoding.n_best = get_or<int>(d, "n_best", cfg.decoding.n_best);
        cfg.decoding.trie_path = get_or<std::string>(d, "trie_path", cfg.decoding.trie_path);
    }

    // runtime
    if (root.contains("runtime")) {
        const json& r = root.at("runtime");
        cfg.runtime.batch_size = get_or<int32_t>(r, "batch_size", cfg.runtime.batch_size);
        cfg.runtime.cache_dtype = get_or<std::string>(r, "cache_dtype", cfg.runtime.cache_dtype);
        cfg.runtime.pre_model_path = get_or<std::string>(r, "pre_model_path", cfg.runtime.pre_model_path);
        cfg.runtime.post_model_path = get_or<std::string>(r, "post_model_path", cfg.runtime.post_model_path);
    }

    if (cfg.pre_model.mhsa_heads <= 0 || cfg.pre_model.attn_dim % cfg.pre_model.mhsa_heads != 0) {
        throw std::runtime_error("ModelPackageConfig::load: pre_model.attn_dim not divisible by mhsa_heads");
    }
    if (cfg.post_model.mhsa_heads <= 0 || cfg.post_model.attn_dim % cfg.post_model.mhsa_heads != 0) {
        throw std::runtime_error("ModelPackageConfig::load: post_model.attn_dim not divisible by mhsa_heads");
    }
    if (cfg.post_model.mhca_heads <= 0 || cfg.post_model.mhca_attn_dim % cfg.post_model.mhca_heads != 0) {
        throw std::runtime_error("ModelPackageConfig::load: post_model.mhca_attn_dim not divisible by mhca_heads");
    }

    return cfg;
}

}  // namespace phono::core
