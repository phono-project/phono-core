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
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace phono::core {

// Errors reported while parsing / validating a runtime core_config against
// the limits of a loaded model package.
enum class CoreConfigError {
    Ok = 0,
    InvalidJson,
    BeamSizeMismatch,          // beam_size does not match the model batch width
    MaxContextLengthExceeded,  // max_context_length + 1 does not fit pre max_seqlen
    MaxPinyinLengthInvalid,    // max_pinyin_length out of range / exceeds post model limit
    MaxHistoryLengthInvalid,   // max_history_length out of range / violates the window invariant
    SlackInvalid,              // slack_interval out of range
    MinAcceptContextInvalid,   // min_accept_context out of range
    DecayInvalid,              // trial_ratio / decay_alpha / decay_lambda out of range
};

const char* core_config_error_name(CoreConfigError error);

class ModelPackageConfig;  // fwd decl, defined below

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

// The on-device runtime (core_config) that drives the inference session and
// the context slot manager. It is user supplied at runtime (e.g. passed to
// the C API as a JSON string) and must be validated against the loaded
// model's hard limits before use.
//
// Window guarantees (see parse_core_config):
//   * max_pinyin_length <= post_model.max_seqlen (hard input limit of post)
//   * max_history_length < max_context_length - max_pinyin_length
//     ensures that a full history window plus a full pinyin window still fit
//     inside max_context_length (i.e. before the committed text hits the
//     pre-model cache limit on the next screen-up).
struct CoreConfig {
    int32_t beam_size = 1;              // must equal the model's beam/batch width
    int32_t slack_interval = 8;         // windowing slack used when evicting history
    int32_t min_accept_context = 8;     // minimum reusable suffix length for slot reuse
    int32_t max_context_length = 127;   // committed context-id capacity per slot
    int32_t max_history_length = 100;   // soft cap on committed history ids (pre model)
    int32_t max_pinyin_length = 32;     // cap on the pinyin window fed to post model
    double trial_ratio = 0.25;
    double decay_alpha = 0.5;
    double decay_lambda = 1.0 / 60.0;
};

// Builds a CoreConfig that is guaranteed valid for the given model (used by
// the convenience InferenceSession constructor).
CoreConfig default_core_config(const ModelPackageConfig& model);

// Parses a core_config JSON object (or string) and validates every field
// against the loaded model's limits. On success fills `out` and returns Ok;
// otherwise returns the first failing CoreConfigError and leaves `out`
// untouched. `json` may be either a JSON object value or a string containing
// a JSON object.
CoreConfigError parse_core_config(const nlohmann::json& json,
                                  const ModelPackageConfig& model,
                                  CoreConfig& out);

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
    static constexpr const char* kSupportedFormatVersion = "2.1";

    // Loads and validates `<package_root>/config.json`.
    static ModelPackageConfig load(const std::string& package_root);

    // Resolves a path from config.json (e.g. "bins/pre_model.pte") against
    // package_root. Absolute input paths are returned unchanged.
    std::string resolve(const std::string& relative_path) const;

    CommonConfig common;
    PreModelDims pre_model;
    PostModelDims post_model;
    VocabPaths vocabs;
    RuntimeParams runtime;

    std::string model_version = "";
    std::string model_format_version;

    std::string package_root;
};

}  // namespace phono::core
