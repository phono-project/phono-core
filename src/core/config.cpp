#include "config.hpp"

#include <algorithm>
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

int32_t model_beam_width(const ModelPackageConfig& model) {
    // The beam/batch width is fixed by the exported pre pass 2 program; when
    // the method metadata is unavailable the package config is the fallback.
    return model.runtime.batch_size;
}

bool is_valid_ratio(double value) { return value > 0.0 && value <= 1.0; }

}  // namespace

const char* core_config_error_name(CoreConfigError error) {
    switch (error) {
        case CoreConfigError::Ok: return "ok";
        case CoreConfigError::InvalidJson: return "invalid_json";
        case CoreConfigError::BeamSizeMismatch: return "beam_size_mismatch";
        case CoreConfigError::MaxContextLengthExceeded: return "max_context_length_exceeded";
        case CoreConfigError::MaxPinyinLengthInvalid: return "max_pinyin_length_invalid";
        case CoreConfigError::MaxHistoryLengthInvalid: return "max_history_length_invalid";
        case CoreConfigError::SlackInvalid: return "slack_interval_invalid";
        case CoreConfigError::MinAcceptContextInvalid: return "min_accept_context_invalid";
        case CoreConfigError::DecayInvalid: return "decay_config_invalid";
    }
    return "unknown";
}

CoreConfig default_core_config(const ModelPackageConfig& model) {
    CoreConfig cfg;
    cfg.beam_size = model_beam_width(model);
    cfg.max_context_length = model.pre_model.max_seqlen - 1;
    cfg.max_pinyin_length = model.post_model.max_seqlen;
    cfg.max_history_length = std::max<int32_t>(
        1, cfg.max_context_length - cfg.max_pinyin_length - 1);
    cfg.slack_interval = std::max<int32_t>(0, std::min<int32_t>(8, cfg.max_history_length - 1));
    return cfg;
}

CoreConfigError parse_core_config(const json& input,
                                  const ModelPackageConfig& model,
                                  CoreConfig& out) {
    json object;
    if (input.is_string()) {
        try {
            object = json::parse(input.get<std::string>());
        } catch (...) {
            return CoreConfigError::InvalidJson;
        }
    } else if (input.is_object()) {
        object = input;
    } else {
        return CoreConfigError::InvalidJson;
    }

    const int32_t pre_max = model.pre_model.max_seqlen;
    const int32_t post_max = model.post_model.max_seqlen;
    const int32_t model_beam = model_beam_width(model);

    // Start from model-derived defaults so an absent key always yields a
    // value that is valid for this model, then overlay the supplied keys.
    CoreConfig candidate = default_core_config(model);
    try {
        candidate.beam_size = object.value("beam_size", candidate.beam_size);
        candidate.slack_interval = object.value("slack_interval", candidate.slack_interval);
        candidate.min_accept_context = object.value("min_accept_context", candidate.min_accept_context);
        candidate.max_context_length = object.value("max_context_length", candidate.max_context_length);
        candidate.max_history_length = object.value("max_history_length", candidate.max_history_length);
        candidate.max_pinyin_length = object.value("max_pinyin_length", candidate.max_pinyin_length);
        candidate.trial_ratio = object.value("trial_ratio", candidate.trial_ratio);
        candidate.decay_alpha = object.value("decay_alpha", candidate.decay_alpha);
        candidate.decay_lambda = object.value("decay_lambda", candidate.decay_lambda);
    } catch (...) {
        return CoreConfigError::InvalidJson;
    }

    if (candidate.beam_size <= 0 || candidate.beam_size != model_beam) {
        return CoreConfigError::BeamSizeMismatch;
    }
    if (candidate.max_context_length < 0 || candidate.max_context_length + 1 > pre_max) {
        return CoreConfigError::MaxContextLengthExceeded;
    }
    if (candidate.max_pinyin_length <= 0 || candidate.max_pinyin_length > post_max) {
        return CoreConfigError::MaxPinyinLengthInvalid;
    }
    if (candidate.max_history_length <= 0 ||
        candidate.max_history_length > candidate.max_context_length ||
        candidate.max_history_length >= candidate.max_context_length - candidate.max_pinyin_length) {
        // max_history_length must stay strictly below
        // max_context_length - max_pinyin_length so that a full history window
        // plus a full pinyin window still fits before the committed text hits
        // the pre-model cache limit.
        return CoreConfigError::MaxHistoryLengthInvalid;
    }
    if (candidate.slack_interval < 0 ||
        candidate.slack_interval >= candidate.max_history_length) {
        return CoreConfigError::SlackInvalid;
    }
    if (candidate.min_accept_context <= 0) {
        return CoreConfigError::MinAcceptContextInvalid;
    }
    if (!is_valid_ratio(candidate.trial_ratio) ||
        candidate.decay_alpha < 0.0 || candidate.decay_lambda < 0.0) {
        return CoreConfigError::DecayInvalid;
    }

    out = candidate;
    return CoreConfigError::Ok;
}

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

    cfg.model_version = get_or<std::string>(root, "model_version", cfg.model_version);
    cfg.model_format_version = get_or<int32_t>(root, "model_format_version", cfg.model_format_version);

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

    // runtime
    if (root.contains("runtime")) {
        const json& r = root.at("runtime");
        cfg.runtime.batch_size = get_or<int32_t>(r, "batch_size", cfg.runtime.batch_size);
        cfg.runtime.cache_dtype = get_or<std::string>(r, "cache_dtype", cfg.runtime.cache_dtype);
        cfg.runtime.pre_model_path = get_or<std::string>(r, "pre_model_path", cfg.runtime.pre_model_path);
        cfg.runtime.post_model_path = get_or<std::string>(r, "post_model_path", cfg.runtime.post_model_path);
        cfg.runtime.pre_pass1_method = get_or<std::string>(r, "pre_pass1_method", cfg.runtime.pre_pass1_method);
        cfg.runtime.pre_pass2_method = get_or<std::string>(r, "pre_pass2_method", cfg.runtime.pre_pass2_method);
        cfg.runtime.post_method = get_or<std::string>(r, "post_method", cfg.runtime.post_method);
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
    if (cfg.runtime.batch_size <= 0) {
        throw std::runtime_error("ModelPackageConfig::load: runtime.batch_size must be positive");
    }

    return cfg;
}

}  // namespace phono::core
