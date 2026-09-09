#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "algo/trie.hpp"
#include "algo/pinyin_segment.hpp"
#include "algo/zh2hans.hpp"
#include "context/context.hpp"
#include "core/config.hpp"
#include "core/text_normalizer.hpp"
#include "core/pinyin_normalizer.hpp"
#include "core/tokenizer.hpp"

namespace {

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("cannot create test file");
    out << contents;
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

phono::core::ModelPackageConfig test_config() {
    phono::core::ModelPackageConfig cfg;
    cfg.pre_model.max_seqlen = 16;
    cfg.pre_model.mhsa_layers = 1;
    cfg.pre_model.mhsa_heads = 1;
    cfg.pre_model.attn_dim = 4;
    cfg.post_model.max_seqlen = 8;
    cfg.post_model.mhca_heads = 1;
    cfg.post_model.mhca_attn_dim = 4;
    cfg.runtime.batch_size = 2;
    return cfg;
}

phono::core::CoreConfig test_core_config() {
    phono::core::CoreConfig cfg;
    cfg.beam_size = 2;
    cfg.slack_interval = 1;
    cfg.min_accept_context = 2;
    cfg.max_context_length = 8;
    cfg.max_history_length = 4;
    cfg.max_pinyin_length = 3;
    return cfg;
}

void test_trie_lookup_and_all_matches() {
    phono::algo::Trie trie;
    trie.insert("w");
    trie.insert("wo");
    trie.insert("xi");
    trie.insert("xian");

    check(trie.contains("w"), "trie should contain terminal prefixes");
    check(trie.contains("wo"), "trie should contain inserted words");
    check(!trie.contains("x"), "trie should reject non-terminal prefixes");
    check(!trie.contains(""), "trie should reject the empty word");
    check(trie.match_lengths("woxian", 0) == std::vector<size_t>({1, 2}),
          "trie should return every terminal match");
    check(trie.match_lengths("woxian", 2) == std::vector<size_t>({2, 4}),
          "trie should return every match from an offset");
    check(trie.match_lengths("xian", 0, 2) == std::vector<size_t>({2}),
          "trie should respect the caller's end boundary");
    check(trie.match_lengths("unknown").empty(), "trie should report a missing match");
    check(trie.match_lengths("wo", 2).empty(), "trie should reject an end offset");
}

void test_pinyin_normalization() {
    phono::core::PinyinNormalizationConfig config;
    const auto normalized = phono::core::normalize_pinyin("Jv'an-LV", config);
    check(normalized.canonical_input == "juan-lv", "normalizer should lowercase and map jv");
    check(normalized.forced_boundaries.size() == normalized.canonical_input.size() - 1,
          "normalizer should align forced boundaries with canonical gaps");
    check(normalized.forced_boundaries[1], "quote should force the preceding canonical gap");
    check(normalized.canonical_input.substr(5) == "lv", "lv must retain v");
    check(normalized.source_offsets[2] == 3, "source offsets should skip separators");

    bool saw_v_to_u = false;
    bool saw_forced = false;
    for (const auto& event : normalized.events) {
        saw_v_to_u = saw_v_to_u || event.type == "v_to_u";
        saw_forced = saw_forced || event.type == "forced_boundary";
    }
    check(saw_v_to_u && saw_forced, "normalizer should report every transformation");

    const auto separated = phono::core::normalize_pinyin("j'v", config);
    check(separated.canonical_input == "jv", "v/u mapping must not cross a forced boundary");
}

void test_gap_viterbi_and_checked_fmm() {
    phono::algo::Trie trie;
    for (const char* token : {"p", "b", "xi", "an", "xian"}) trie.insert(token);

    const auto pb = phono::algo::decode_gap_viterbi(
        "pb", trie, {false}, {-100.0f}, true);
    check(pb.reachable && pb.invalid_char_count == 0 && pb.edges.size() == 2,
          "a fully legal jianpin path must beat invalid fallback edges");

    const auto pv = phono::algo::decode_gap_viterbi(
        "pv", trie, {false}, {0.0f}, true);
    check(pv.reachable && pv.invalid_char_count == 1 && pv.edges.size() == 2 &&
              pv.edges[1].invalid,
          "unavoidable invalid characters should remain explicitly marked");

    const auto strict_pv = phono::algo::decode_gap_viterbi(
        "pv", trie, {false}, {0.0f}, false);
    check(!strict_pv.reachable, "strict DAG decoding should reject an incomplete path");

    const auto joined = phono::algo::decode_gap_viterbi(
        "xian", trie, {false, false, false}, {0.0f, -2.0f, 0.0f}, false);
    check(joined.edges.size() == 1, "negative boundary logits should prefer xian");
    const auto split = phono::algo::decode_gap_viterbi(
        "xian", trie, {false, false, false}, {0.0f, 2.0f, 0.0f}, false);
    check(split.edges.size() == 2 && split.edges[0].end == 2,
          "positive boundary logits should prefer xi-an");
    const auto forced = phono::algo::decode_gap_viterbi(
        "xian", trie, {false, true, false}, {0.0f, -100.0f, 0.0f}, false);
    check(forced.edges.size() == 2, "forced boundaries must prohibit crossing edges");

    const auto fmm = phono::algo::separate_fmm_checked("xipv", trie, {false, false, false});
    check(fmm.invalid_char_count == 1 && fmm.edges.back().invalid,
          "checked FMM must report rather than silently accept invalid characters");

    phono::algo::Trie lookahead;
    for (const char* token : {"a", "ab", "bc"}) lookahead.insert(token);
    const auto complete = phono::algo::separate_fmm_checked(
        "abc", lookahead, {false, false});
    check(complete.invalid_char_count == 0 && complete.edges.size() == 2 &&
              complete.edges.front().end == 1,
          "checked FMM must not take a longest prefix that creates an avoidable dead end");
}

void test_tokenizer_normalizes_and_skips_unknown() {
    const auto dir = std::filesystem::temp_directory_path() / "phono_core_tokenizer_test";
    std::filesystem::create_directories(dir);
    write_file(dir / "chinese.txt", "\xE5\x8F\xB0\nA\n");
    write_file(dir / "context.txt", "\xE5\x8F\xB0\nA\nknown\n");
    write_file(dir / "pinyin.txt", "w\nwo\nx\nxi\nxian\na\nan\nh\nhuan\nn\nni\n");

    phono::core::Tokenizer tokenizer(
        (dir / "chinese.txt").string(), (dir / "context.txt").string(),
        (dir / "pinyin.txt").string(), {"bos_token"});

    // U+81FA (traditional Taiwan character) -> U+53F0 and full-width A -> A.
    const auto ids = tokenizer.encode_context("\xE8\x87\xBA\xEF\xBC\xA1X");
    check(ids.size() == 2, "normalization should produce two known ids");
    check(ids[0] == 0, "traditional character should simplify");
    check(ids[1] == 1, "full-width character should be NFKC normalized");
    check(tokenizer.encode_context("X").empty(), "unknown context ids should be skipped");
    check(tokenizer.chinese_id_to_context_id(0) == 0, "Chinese/context ids should map");
    check(tokenizer.chinese_id_to_context_id(99) == -1, "unknown Chinese id should fail");
    check(tokenizer.find_pinyin_id_exact("wo").has_value(),
          "exact pinyin lookup should find vocabulary entries");
    check(!tokenizer.find_pinyin_id_exact("invalid").has_value(),
          "exact pinyin lookup must not silently repair unknown entries");
    check(tokenizer.find_pinyin_id_nearest("wo") == *tokenizer.find_pinyin_id_exact("wo"),
          "nearest lookup should preserve exact entries");
    check(tokenizer.pinyin_token(*tokenizer.find_pinyin_id_exact("wo")) == "wo",
          "pinyin ids should map back to vocabulary tokens");
    check(tokenizer.pinyin_token(tokenizer.find_pinyin_id_nearest("v")) == "a",
          "nearest-token ties should use lexical order for deterministic repair");
    check(tokenizer.pinyin_trie().contains("xian"),
          "tokenizer should expose its immutable pinyin trie to segmentation algorithms");

    std::filesystem::remove_all(dir);
}

void test_zh2hans_simplification() {
    using phono::algo::to_simplified_zh;
    using phono::core::normalize_text_utf8;

    // 臺 -> 台 (single-character rule).
    check(to_simplified_zh("\xE8\x87\xBA") == "\xE5\x8F\xB0", "臺 should simplify to 台");
    // 乾 -> 干 (single-character rule).
    check(to_simplified_zh("\xE4\xB9\xBE") == "\xE5\xB9\xB2", "乾 should simplify to 干");
    // 一釐 -> 一厘 (phrase rule).
    check(to_simplified_zh("\xE4\xB8\x80\xE9\x87\x90") == "\xE4\xB8\x80\xE5\x8E\x98",
          "一釐 should simplify to 一厘");
    // 上昇 -> 上升 (phrase rule).
    check(to_simplified_zh("\xE4\xB8\x8A\xE6\x98\x87") == "\xE4\xB8\x8A\xE5\x8D\x87",
          "上昇 should simplify to 上升");
    // Maximal munch: the identity phrase rule 《易乾 shields 乾 from the
    // single-character rule, so it must NOT become 《易干.
    check(to_simplified_zh("\xE3\x80\x8A\xE6\x98\x93\xE4\xB9\xBE") ==
              "\xE3\x80\x8A\xE6\x98\x93\xE4\xB9\xBE",
          "《易乾 identity phrase should shield 乾");
    // Fallback to per-character conversion when no phrase matches: 臺北 -> 台北.
    check(to_simplified_zh("\xE8\x87\xBA\xE5\x8C\x97") == "\xE5\x8F\xB0\xE5\x8C\x97",
          "臺北 should simplify character by character");
    // ASCII passes through untouched.
    check(to_simplified_zh("Hello \xE8\x87\xBA") == "Hello \xE5\x8F\xB0",
          "ASCII should pass through unchanged");
    check(to_simplified_zh("").empty(), "empty input should stay empty");

    // Full normalization: traditional char simplified, full-width A -> A.
    check(normalize_text_utf8("\xE8\x87\xBA\xEF\xBC\xA1X") == "\xE5\x8F\xB0"
                                                              "AX",
          "normalize_text_utf8 should simplify and NFKC-normalize");
    check(normalize_text_utf8("").empty(), "empty normalization input should stay empty");
}

void test_cache_slice_operations() {
    auto cache = phono::context::make_zero_persistent_tensor({1, 2, 2, 8, 1, 1});
    for (int32_t kv = 0; kv < 2; ++kv) {
        for (int32_t batch = 0; batch < 2; ++batch) {
            for (int32_t pos = 0; pos < 8; ++pos) {
                cache.data()[(kv * 2 + batch) * 8 + pos] =
                    static_cast<float>(100 * kv + 10 * batch + pos);
            }
        }
    }
    cache.shift_batch_tokens(0, 2, 3);
    check(cache.data()[0] == 0.0f, "BOS is protected");
    check(cache.data()[1] == 3.0f, "shift should copy source position 3");
    check(cache.data()[2] == 4.0f, "shift should preserve order");
    check(cache.data()[3] == 5.0f, "shift should preserve retained tokens");

    cache.copy_batch_slice_to_all(0, 1, 3);
    check(cache.data()[8 + 1] == 3.0f, "slice should copy to beam one");
    check(cache.data()[8 + 2] == 4.0f, "slice should copy to beam one");
    check(cache.data()[8 + 3] == 5.0f, "slice should copy to beam one");

    cache.reorder_batches({1, 0});
    check(cache.data()[1] == 3.0f, "reorder should preserve selected beam");
    check(cache.data()[8 + 1] == 3.0f, "reorder should preserve selected beam");

    // Slice reordering supports duplicated parents while preserving both the
    // committed history before the slice and dirty future positions after it.
    for (int32_t batch = 0; batch < 2; ++batch) {
        for (int32_t pos = 0; pos < 8; ++pos) {
            cache.data()[batch * 8 + pos] = static_cast<float>(10 * batch + pos);
        }
    }
    std::vector<float_t> reorder_scratch;
    cache.reorder_batch_slice({1, 1}, 2, 3, reorder_scratch);
    check(cache.data()[0] == 0.0f, "slice reorder must preserve history");
    check(cache.data()[2] == 12.0f, "slice reorder should select parent data");
    check(cache.data()[8 + 4] == 14.0f, "slice reorder should duplicate a parent");
    check(cache.data()[5] == 5.0f, "slice reorder must preserve dirty future data");
}

void test_context_auto_matching() {
    const auto cfg = test_config();
    const phono::core::CoreConfig core = test_core_config();
    phono::context::ContextManager manager(cfg, 2, core);
    auto& slot = manager.get_context_by_id(0);
    const int32_t ids[] = {1, 2, 3, 4};
    slot.append_context_ids(ids, 4);
    slot.set_history_seqlen(5);
    slot.set_current_position(5);
    const int32_t cache_max = slot.self_kv.max_seqlen();
    const int64_t token_stride = slot.self_kv.token_stride_elements();
    const int64_t batch_stride = slot.self_kv.batch_stride_elements();
    for (int32_t kv = 0; kv < 2; ++kv) {
        for (int32_t batch = 0; batch < 2; ++batch) {
            for (int32_t pos = 0; pos < cache_max; ++pos) {
                const int64_t offset = (kv * 2 + batch) * batch_stride + pos * token_stride;
                for (int64_t d = 0; d < token_stride; ++d) {
                    slot.self_kv.data()[offset + d] = static_cast<float>(pos);
                }
            }
        }
    }

    slot.shift_cached_tokens(2, 2);
    check(slot.self_kv.data()[token_stride] == 3.0f,
          "direct cache shift should preserve first match");
    slot.self_kv.zero_();
    for (int32_t kv = 0; kv < 2; ++kv) {
        for (int32_t batch = 0; batch < 2; ++batch) {
            for (int32_t pos = 0; pos < cache_max; ++pos) {
                const int64_t offset = (kv * 2 + batch) * batch_stride + pos * token_stride;
                for (int64_t d = 0; d < token_stride; ++d) {
                    slot.self_kv.data()[offset + d] = static_cast<float>(pos);
                }
            }
        }
    }
    auto* shifted = manager.get_context_auto({3, 4, 5});
    check(shifted == &slot, "middle match should reuse the existing slot");
    check(shifted->context_ids_len() == 2, "middle match should retain matched ids");
    check(shifted->context_ids()[0] == 3, "matched id should move after BOS");
    check(shifted->context_ids()[1] == 4, "matched id should move after BOS");
    check(shifted->history_seqlen() == 3, "history should include BOS");
    check(shifted->current_position() == 3, "current length should include BOS");
    check(shifted->self_kv.data()[token_stride] == 3.0f,
          "cache shift should preserve first match");
    check(shifted->self_kv.data()[2 * token_stride] == 4.0f,
          "cache shift should preserve second match");
    check(shifted->self_kv.data()[batch_stride + token_stride] == 3.0f,
          "cache shift should affect every beam");

    auto* truncated = manager.get_context_auto({3});
    check(truncated == &slot, "prefix match should reuse the slot");
    check(truncated->context_ids_len() == 1, "prefix match should truncate ids");
    check(truncated->history_seqlen() == 2, "prefix match should truncate history");
    check(truncated->current_position() == 2, "prefix match should truncate current length");

    auto* divergent = manager.get_context_auto({3, 9});
    check(divergent == &slot, "common prefix should survive a tail edit");
    check(divergent->context_ids_len() == 1, "tail edit should retain the common prefix");
    check(divergent->history_seqlen() == 2, "tail edit should reset cached history length");
}

void test_core_config_default_and_parse() {
    const auto cfg = test_config();
    const phono::core::CoreConfig defaults = phono::core::default_core_config(cfg);
    check(defaults.beam_size == 2, "default beam should match model batch");
    check(defaults.max_context_length == 15, "default context should be pre_max - 1");
    check(defaults.max_pinyin_length == 8, "default pinyin cap should match post_max");
    check(defaults.max_history_length < defaults.max_context_length - defaults.max_pinyin_length,
          "default config must satisfy the window invariant");
    check(defaults.slack_interval < defaults.max_history_length,
          "default slack must leave a positive window");

    // The new core_config key names supersede the old N / T.
    const nlohmann::json options = {
        {"schema_version", "1.0"},
        {"beam_size", 2},
        {"slack_interval", 1},
        {"min_accept_context", 2},
        {"max_context_length", 8},
        {"max_history_length", 4},
        {"max_pinyin_length", 3},
    };
    phono::core::CoreConfig parsed;
    check(phono::core::parse_core_config(options, cfg, parsed) ==
              phono::core::CoreConfigError::Ok,
          "valid core config should parse");
    check(parsed.slack_interval == 1, "slack_interval should be read from JSON");
    check(parsed.min_accept_context == 2, "min_accept_context should be read from JSON");
    check(parsed.max_history_length == 4, "max_history_length should be read from JSON");
    check(parsed.max_pinyin_length == 3, "max_pinyin_length should be read from JSON");

    // Legacy N / T keys must no longer configure anything.
    const nlohmann::json legacy = {
        {"schema_version", "1.0"},
        {"beam_size", 2}, {"N", 1}, {"T", 2},
        {"max_context_length", 15}, {"max_history_length", 10}, {"max_pinyin_length", 3},
    };
    phono::core::CoreConfig legacy_parsed;
    check(phono::core::parse_core_config(legacy, cfg, legacy_parsed) ==
              phono::core::CoreConfigError::Ok,
          "legacy N/T keys should be ignored, not rejected");
    check(legacy_parsed.slack_interval == defaults.slack_interval,
          "legacy N key must not set slack_interval");
    check(legacy_parsed.min_accept_context == defaults.min_accept_context,
          "legacy T key must not set min_accept_context");
    check(legacy_parsed.slack_interval != 1 && legacy_parsed.min_accept_context != 2,
          "legacy N/T values must not leak into the parsed config");
}

void test_model_format_version() {
    const auto dir = std::filesystem::temp_directory_path() / "phono_core_format_test";
    std::filesystem::create_directories(dir);

    write_file(dir / "config.json", R"({"model_format_version":"2.2"})");
    const auto valid = phono::core::ModelPackageConfig::load(dir.string());
    check(valid.model_format_version == "2.2", "v2.2 JSON format should load");

    write_file(dir / "config.json", R"({
        "model_format_version":"2.2",
        "segmenter":{"layout":"BHWC","min_input_chars":3,
                     "max_input_chars":128,"quantization":"none"}
    })");
    const auto with_segmenter = phono::core::ModelPackageConfig::load(dir.string());
    check(with_segmenter.segmenter.has_value(), "JSON segmenter config should load");
    check(with_segmenter.segmenter->quantization == "none",
          "segmenter quantization metadata should be retained");

    for (const std::string& value : {"2", "\"2.1\"", "null"}) {
        write_file(dir / "config.json", "{\"model_format_version\":" + value + "}");
        bool rejected = false;
        try {
            static_cast<void>(phono::core::ModelPackageConfig::load(dir.string()));
        } catch (const std::exception&) {
            rejected = true;
        }
        check(rejected, "non-v2.2 model format should be rejected");
    }
    std::filesystem::remove_all(dir);
}

nlohmann::json to_json(const phono::core::CoreConfig& config) {
    return {
        {"schema_version", "1.0"},
        {"beam_size", config.beam_size},
        {"slack_interval", config.slack_interval},
        {"min_accept_context", config.min_accept_context},
        {"max_context_length", config.max_context_length},
        {"max_history_length", config.max_history_length},
        {"max_pinyin_length", config.max_pinyin_length},
        {"trial_ratio", config.trial_ratio},
        {"decay_alpha", config.decay_alpha},
        {"decay_lambda", config.decay_lambda},
    };
}

void test_core_config_validation() {
    const auto cfg = test_config();
    const phono::core::CoreConfig valid = test_core_config();

    phono::core::CoreConfig out;

    // max_pinyin_length may not exceed the post model's hard limit.
    auto too_long_pinyin = valid;
    too_long_pinyin.max_pinyin_length = cfg.post_model.max_seqlen + 1;
    check(phono::core::parse_core_config(to_json(too_long_pinyin), cfg, out) ==
              phono::core::CoreConfigError::MaxPinyinLengthInvalid,
          "max_pinyin_length beyond post hard limit should be rejected");

    // max_context_length must fit the pre model cache including BOS.
    auto too_long_context = valid;
    too_long_context.max_context_length = cfg.pre_model.max_seqlen;  // pre_max - 1 is the max
    check(phono::core::parse_core_config(to_json(too_long_context), cfg, out) ==
              phono::core::CoreConfigError::MaxContextLengthExceeded,
          "max_context_length must stay below pre_model.max_seqlen");

    // The window invariant max_history_length < max_context_length -
    // max_pinyin_length must hold, otherwise the committed text could overflow
    // before the next screen-up.
    auto bad_invariant = valid;
    bad_invariant.max_history_length = bad_invariant.max_context_length -
                                       bad_invariant.max_pinyin_length;  // not strictly less
    check(phono::core::parse_core_config(to_json(bad_invariant), cfg, out) ==
              phono::core::CoreConfigError::MaxHistoryLengthInvalid,
          "max_history_length must be strictly less than max_context_length - max_pinyin_length");
    auto bad_invariant2 = valid;
    bad_invariant2.max_history_length = bad_invariant2.max_context_length;  // > window
    check(phono::core::parse_core_config(to_json(bad_invariant2), cfg, out) ==
              phono::core::CoreConfigError::MaxHistoryLengthInvalid,
          "max_history_length above max_context_length should be rejected");

    // slack_interval must leave a non-empty window.
    auto bad_slack = valid;
    bad_slack.slack_interval = bad_slack.max_history_length;
    check(phono::core::parse_core_config(to_json(bad_slack), cfg, out) ==
              phono::core::CoreConfigError::SlackInvalid,
          "slack_interval must be strictly below max_history_length");

    // beam_size must match the model's beam/batch width.
    auto bad_beam = valid;
    bad_beam.beam_size = valid.beam_size + 1;
    check(phono::core::parse_core_config(to_json(bad_beam), cfg, out) ==
              phono::core::CoreConfigError::BeamSizeMismatch,
          "beam_size mismatch should be rejected");

    // A JSON string is accepted.
    check(phono::core::parse_core_config(nlohmann::json("not json"), cfg, out) ==
              phono::core::CoreConfigError::InvalidJson,
          "malformed JSON string should be rejected");
    check(phono::core::parse_core_config(
              nlohmann::json::object({}), cfg, out) ==
              phono::core::CoreConfigError::UnsupportedSchemaVersion,
          "unversioned core config should be rejected");
}

void test_engine_config_parse() {
    phono::core::EngineConfig config;
    const nlohmann::json valid = {
        {"schema_version", "1.0"},
        {"tokenizer", {
            {"normalization", {
                {"lowercase_ascii", false},
                {"normalize_v_to_u", true},
                {"separators", "' "},
            }},
            {"segment_mode", "strict"},
            {"repair", true},
            {"max_pinyin_chars", 64},
        }},
    };
    check(phono::core::parse_engine_config(valid, config) ==
              phono::core::EngineConfigError::Ok,
          "valid engine config should parse");
    check(config.tokenizer.segment_mode == phono::core::SegmentMode::Strict &&
              config.tokenizer.repair && config.tokenizer.max_pinyin_chars == 64 &&
              !config.tokenizer.normalization.lowercase_ascii,
          "engine tokenizer options should be preserved");

    auto bad_mode = valid;
    bad_mode["tokenizer"]["segment_mode"] = "guess";
    check(phono::core::parse_engine_config(bad_mode, config) ==
              phono::core::EngineConfigError::InvalidSegmentMode,
          "unknown segmentation mode should be rejected");
    auto bad_limit = valid;
    bad_limit["tokenizer"]["max_pinyin_chars"] = 0;
    check(phono::core::parse_engine_config(bad_limit, config) ==
              phono::core::EngineConfigError::MaxPinyinCharsInvalid,
          "non-positive character limits should be rejected");
    auto bad_separator = valid;
    bad_separator["tokenizer"]["normalization"]["separators"] = "a";
    check(phono::core::parse_engine_config(bad_separator, config) ==
              phono::core::EngineConfigError::InvalidTokenizerConfig,
          "letters cannot be configured as separators");
}

}  // namespace

int main() {
    test_trie_lookup_and_all_matches();
    test_pinyin_normalization();
    test_gap_viterbi_and_checked_fmm();
    test_tokenizer_normalizes_and_skips_unknown();
    test_zh2hans_simplification();
    test_cache_slice_operations();
    test_context_auto_matching();
    test_core_config_default_and_parse();
    test_model_format_version();
    test_core_config_validation();
    test_engine_config_parse();
    return 0;
}
