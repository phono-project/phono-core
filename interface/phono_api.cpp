// phono_api.cpp
//
// C-ABI wrapper around phono-core. All C++ exceptions raised by the core are
// caught here and mapped to phono_status error codes; no exception ever
// crosses the ABI boundary.

#include "phono_api.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/config.hpp"
#include "engine/inference_engine.hpp"

namespace {

using phono::core::CoreConfig;
using phono::core::CoreConfigError;
using phono::core::EngineConfig;
using phono::core::EngineConfigError;
using phono::core::parse_core_config;
using phono::core::parse_engine_config;
using phono::engine::GenerateResult;
using phono::engine::InferenceError;
using phono::engine::InferenceEngine;
using phono::engine::InferenceSession;

thread_local std::string g_last_error;

void set_last_error(const std::string& message) { g_last_error = message; }

phono_status map_inference_error(InferenceError error) {
    switch (error) {
        case InferenceError::Ok: return PHONO_OK;
        case InferenceError::InvalidArgument: return PHONO_INVALID_ARGUMENT;
        case InferenceError::ContextLimitExceeded: return PHONO_CONTEXT_LIMIT_EXCEEDED;
        case InferenceError::PinyinLimitExceeded: return PHONO_PINYIN_LIMIT_EXCEEDED;
        case InferenceError::InvalidPinyin: return PHONO_INVALID_PINYIN;
        case InferenceError::NoCandidates: return PHONO_NO_CANDIDATES;
        case InferenceError::Cancelled: return PHONO_CANCELLED;
        case InferenceError::ModelError: return PHONO_MODEL_ERROR;
    }
    return PHONO_MODEL_ERROR;
}

phono_status map_config_error(CoreConfigError error) {
    return error == CoreConfigError::Ok ? PHONO_OK : PHONO_CONFIG_ERROR;
}

// Bridges the C ABI int-returning callback to the engine's bool-returning
// callback. `user_data` points to a CancelBridge.
struct CancelBridge {
    phono_cancel_fn fn;
    void* data;
};

bool cancellation_bridge(void* user_data) {
    const auto* bridge = static_cast<const CancelBridge*>(user_data);
    return bridge != nullptr && bridge->fn != nullptr && bridge->fn(bridge->data) != 0;
}

// Parses a core_config JSON string into `out`, reporting PHONO_CONFIG_ERROR
// with a diagnostic message on failure.
phono_status parse_core_config_string(InferenceEngine& engine, const char* core_config_json,
                                      CoreConfig& out) {
    if (core_config_json == nullptr || *core_config_json == '\0') {
        set_last_error("core_config_json is empty");
        return PHONO_CONFIG_ERROR;
    }
    nlohmann::json options;
    try {
        options = nlohmann::json::parse(core_config_json);
    } catch (const std::exception& e) {
        set_last_error(std::string("core_config is not valid JSON: ") + e.what());
        return PHONO_CONFIG_ERROR;
    }
    const CoreConfigError error = parse_core_config(options, engine.config(), out);
    if (error != CoreConfigError::Ok) {
        set_last_error(std::string("core_config rejected: ") +
                       phono::core::core_config_error_name(error));
        return PHONO_CONFIG_ERROR;
    }
    return PHONO_OK;
}

phono_status parse_engine_config_string(const char* engine_config_json, EngineConfig& out) {
    if (engine_config_json == nullptr || *engine_config_json == '\0') {
        set_last_error("engine_config_json is empty");
        return PHONO_CONFIG_ERROR;
    }
    nlohmann::json options;
    try {
        options = nlohmann::json::parse(engine_config_json);
    } catch (const std::exception& e) {
        set_last_error(std::string("engine_config is not valid JSON: ") + e.what());
        return PHONO_CONFIG_ERROR;
    }
    const EngineConfigError error = parse_engine_config(options, out);
    if (error != EngineConfigError::Ok) {
        set_last_error(std::string("engine_config rejected: ") +
                       phono::core::engine_config_error_name(error));
        return PHONO_CONFIG_ERROR;
    }
    return PHONO_OK;
}

char* copy_string(const std::string& value) {
    char* buffer = static_cast<char*>(std::malloc(value.size() + 1));
    if (buffer != nullptr) std::memcpy(buffer, value.c_str(), value.size() + 1);
    return buffer;
}

// Convenience: builds the C++ view of a token-id vector into the caller's
// malloc'd buffer.
phono_status copy_ids(const std::vector<int32_t>& ids, int32_t** out_ids, int32_t* out_count) {
    if (out_ids == nullptr || out_count == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    *out_ids = nullptr;
    *out_count = 0;
    if (ids.empty()) {
        return PHONO_OK;
    }
    auto* buffer = static_cast<int32_t*>(std::malloc(ids.size() * sizeof(int32_t)));
    if (buffer == nullptr) {
        return PHONO_MODEL_ERROR;
    }
    std::memcpy(buffer, ids.data(), ids.size() * sizeof(int32_t));
    *out_ids = buffer;
    *out_count = static_cast<int32_t>(ids.size());
    return PHONO_OK;
}

}  // namespace

extern "C" {

// ── engine ────────────────────────────────────────────────────────────────

phono_status phono_engine_create(const char* model_package_dir, const char* engine_config_json,
                                 phono_engine** out_engine) {
    if (out_engine == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    *out_engine = nullptr;
    if (model_package_dir == nullptr || *model_package_dir == '\0') {
        set_last_error("model_package_dir is empty");
        return PHONO_INVALID_ARGUMENT;
    }
    EngineConfig engine_config;
    const phono_status config_status =
        parse_engine_config_string(engine_config_json, engine_config);
    if (config_status != PHONO_OK) return config_status;
    try {
        auto* engine = new InferenceEngine(model_package_dir, std::move(engine_config));
        *out_engine = reinterpret_cast<phono_engine*>(engine);
        return PHONO_OK;
    } catch (const std::invalid_argument& e) {
        set_last_error(std::string("engine_config rejected: ") + e.what());
        return PHONO_CONFIG_ERROR;
    } catch (const std::exception& e) {
        set_last_error(std::string("failed to load model package: ") + e.what());
        return PHONO_MODEL_ERROR;
    } catch (...) {
        set_last_error("failed to load model package: unknown error");
        return PHONO_MODEL_ERROR;
    }
}

void phono_engine_destroy(phono_engine* engine) {
    delete reinterpret_cast<InferenceEngine*>(engine);
}

int32_t phono_engine_pre_max_seqlen(const phono_engine* engine) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    return e == nullptr ? 0 : e->config().pre_model.max_seqlen;
}

int32_t phono_engine_post_max_seqlen(const phono_engine* engine) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    return e == nullptr ? 0 : e->config().post_model.max_seqlen;
}

int32_t phono_engine_beam_size(const phono_engine* engine) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    return e == nullptr ? 0
                        : (e->pre_pass2_batch_size() > 0 ? e->pre_pass2_batch_size()
                                                         : e->config().runtime.batch_size);
}

char* phono_engine_info_json(const phono_engine* engine) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    if (e == nullptr) return nullptr;
    nlohmann::json info = {
        {"schema_version", "1.0"},
        {"model_version", e->config().model_version},
        {"model_format_version", e->config().model_format_version},
        {"segmenter", {
            {"available", e->smart_segmenter_available()},
            {"fallback", e->smart_segmenter_available() ? nullptr : nlohmann::json("checked_fmm")},
        }},
        {"diagnostics", e->diagnostics()},
    };
    return copy_string(info.dump());
}

phono_status phono_engine_segment_pinyin(const phono_engine* engine,
                                         const char* request_json,
                                         char** out_result_json) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    if (e == nullptr || request_json == nullptr || out_result_json == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    *out_result_json = nullptr;
    nlohmann::json request;
    try {
        request = nlohmann::json::parse(request_json);
        if (!request.is_object() ||
            request.value("schema_version", std::string()) != "1.0" ||
            !request.contains("input") || !request.at("input").is_string()) {
            set_last_error("segmentation request must use schema_version 1.0 and string input");
            return PHONO_CONFIG_ERROR;
        }
    } catch (const std::exception& error) {
        set_last_error(std::string("segmentation request is not valid JSON: ") + error.what());
        return PHONO_CONFIG_ERROR;
    }

    const auto segmented = e->segment_pinyin(request.at("input").get<std::string>());
    const phono_status status = map_inference_error(segmented.error);
    nlohmann::json ranges = nlohmann::json::array();
    for (const auto& range : segmented.invalid_ranges) {
        ranges.push_back({
            {"begin", range.begin}, {"end", range.end}, {"text", range.text},
            {"replacement", range.replacement},
        });
    }
    nlohmann::json events = nlohmann::json::array();
    for (const auto& event : segmented.normalization_events) {
        events.push_back({
            {"type", event.type}, {"begin", event.begin}, {"end", event.end},
            {"before", event.before}, {"after", event.after},
        });
    }
    std::vector<int32_t> pinyin_ids;
    pinyin_ids.reserve(segmented.segments.size());
    for (const auto& token : segmented.segments) {
        const auto id = e->tokenizer().find_pinyin_id_exact(token);
        if (!id) {
            set_last_error("internal segmentation result is absent from pinyin vocabulary");
            return PHONO_MODEL_ERROR;
        }
        pinyin_ids.push_back(*id);
    }
    nlohmann::json response = {
        {"schema_version", "1.0"},
        {"status", {{"ok", status == PHONO_OK}, {"code", phono_error_name(status)}}},
        {"input", segmented.input},
        {"canonical_input", segmented.canonical_input},
        {"normalized_input", segmented.normalized_input},
        {"strategy", segmented.strategy},
        {"segments", segmented.segments},
        {"pinyin_ids", pinyin_ids},
        {"invalid_ranges", std::move(ranges)},
        {"normalization_events", std::move(events)},
    };
    *out_result_json = copy_string(response.dump());
    if (*out_result_json == nullptr) return PHONO_MODEL_ERROR;
    if (status != PHONO_OK) {
        set_last_error(std::string("pinyin segmentation failed: ") + phono_error_name(status));
    }
    return status;
}

// ── tokenizer ─────────────────────────────────────────────────────────────

phono_status phono_tokenizer_encode_context(const phono_engine* engine, const char* text_utf8,
                                            int32_t** out_ids, int32_t* out_count) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    if (e == nullptr || out_ids == nullptr || out_count == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    const std::string text = text_utf8 == nullptr ? std::string() : std::string(text_utf8);
    return copy_ids(e->tokenizer().encode_context(text), out_ids, out_count);
}

int32_t phono_tokenizer_find_pinyin_id_exact(const phono_engine* engine,
                                             const char* token) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    if (e == nullptr || token == nullptr) return -1;
    const auto id = e->tokenizer().find_pinyin_id_exact(token);
    return id ? *id : -1;
}

char* phono_tokenizer_decode(const phono_engine* engine, const int32_t* ids, int32_t count) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    if (e == nullptr || count < 0 || (count > 0 && ids == nullptr)) {
        return nullptr;
    }
    const std::vector<int32_t> id_vec(ids, ids + count);
    const std::string text = e->tokenizer().ids_to_text(id_vec);
    char* buffer = static_cast<char*>(std::malloc(text.size() + 1));
    if (buffer == nullptr) {
        return nullptr;
    }
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return buffer;
}

int32_t phono_tokenizer_chinese_to_context(const phono_engine* engine, int32_t chinese_id) {
    const auto* e = reinterpret_cast<const InferenceEngine*>(engine);
    return e == nullptr ? -1 : e->tokenizer().chinese_id_to_context_id(chinese_id);
}

// ── context manager ───────────────────────────────────────────────────────

phono_status phono_context_manager_create(phono_engine* engine, const char* core_config_json,
                                          int32_t num_contexts,
                                          phono_context_manager** out_manager) {
    auto* e = reinterpret_cast<InferenceEngine*>(engine);
    if (e == nullptr || out_manager == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    *out_manager = nullptr;
    CoreConfig config;
    const phono_status status = parse_core_config_string(*e, core_config_json, config);
    if (status != PHONO_OK) {
        return status;
    }
    try {
        auto* manager = new phono::context::ContextManager(e->config(), num_contexts, config);
        *out_manager = reinterpret_cast<phono_context_manager*>(manager);
        return PHONO_OK;
    } catch (const std::exception& exc) {
        set_last_error(std::string("failed to create context manager: ") + exc.what());
        return PHONO_CONFIG_ERROR;
    }
}

void phono_context_manager_destroy(phono_context_manager* manager) {
    delete reinterpret_cast<phono::context::ContextManager*>(manager);
}

int32_t phono_context_manager_num_contexts(const phono_context_manager* manager) {
    const auto* cm = reinterpret_cast<const phono::context::ContextManager*>(manager);
    return cm == nullptr ? 0 : static_cast<int32_t>(cm->num_contexts());
}

phono_context* phono_context_manager_get_by_id(phono_context_manager* manager, int32_t id) {
    auto* cm = reinterpret_cast<phono::context::ContextManager*>(manager);
    if (cm == nullptr || id < 0 || static_cast<size_t>(id) >= cm->num_contexts()) {
        return nullptr;
    }
    return reinterpret_cast<phono_context*>(&cm->get_context_by_id(id));
}

phono_context* phono_context_manager_get_auto(phono_context_manager* manager,
                                              const int32_t* context_ids, int32_t context_count) {
    auto* cm = reinterpret_cast<phono::context::ContextManager*>(manager);
    if (cm == nullptr || context_count < 0 || (context_count > 0 && context_ids == nullptr)) {
        return nullptr;
    }
    const std::vector<int32_t> ids(context_ids, context_ids + context_count);
    try {
        return reinterpret_cast<phono_context*>(cm->get_context_auto(ids));
    } catch (const std::exception& e) {
        set_last_error(std::string("get_context_auto failed: ") + e.what());
        return nullptr;
    }
}

// ── context accessors ─────────────────────────────────────────────────────

int32_t phono_context_ids_len(const phono_context* context) {
    const auto* ctx = reinterpret_cast<const phono::context::Context*>(context);
    return ctx == nullptr ? 0 : ctx->context_ids_len();
}

int32_t phono_context_current_seqlen(const phono_context* context) {
    const auto* ctx = reinterpret_cast<const phono::context::Context*>(context);
    return ctx == nullptr ? 0 : ctx->current_seqlen();
}

int32_t phono_context_history_seqlen(const phono_context* context) {
    const auto* ctx = reinterpret_cast<const phono::context::Context*>(context);
    return ctx == nullptr ? 0 : ctx->history_seqlen();
}

// ── session ───────────────────────────────────────────────────────────────

phono_status phono_session_create(phono_engine* engine, const char* core_config_json,
                                  phono_session** out_session) {
    auto* e = reinterpret_cast<InferenceEngine*>(engine);
    if (e == nullptr || out_session == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    *out_session = nullptr;
    CoreConfig config;
    const phono_status status = parse_core_config_string(*e, core_config_json, config);
    if (status != PHONO_OK) {
        return status;
    }
    try {
        auto* session = new InferenceSession(*e, config);
        *out_session = reinterpret_cast<phono_session*>(session);
        return PHONO_OK;
    } catch (const std::exception& exc) {
        set_last_error(std::string("failed to create session: ") + exc.what());
        return PHONO_CONFIG_ERROR;
    }
}

void phono_session_destroy(phono_session* session) {
    delete reinterpret_cast<InferenceSession*>(session);
}

int32_t phono_session_beam_size(const phono_session* session) {
    const auto* s = reinterpret_cast<const InferenceSession*>(session);
    return s == nullptr ? 0 : s->beam_size();
}

phono_status phono_session_reset(phono_session* session, phono_context* context) {
    auto* s = reinterpret_cast<InferenceSession*>(session);
    auto* ctx = reinterpret_cast<phono::context::Context*>(context);
    if (s == nullptr || ctx == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    try {
        return map_inference_error(s->reset(*ctx));
    } catch (const std::exception& e) {
        set_last_error(std::string("session reset failed: ") + e.what());
        return PHONO_MODEL_ERROR;
    }
}

phono_status phono_session_fill(phono_session* session, phono_context* context,
                                const int32_t* new_ids, int32_t new_count) {
    auto* s = reinterpret_cast<InferenceSession*>(session);
    auto* ctx = reinterpret_cast<phono::context::Context*>(context);
    if (s == nullptr || ctx == nullptr || new_count < 0 ||
        (new_count > 0 && new_ids == nullptr)) {
        return PHONO_INVALID_ARGUMENT;
    }
    const std::vector<int32_t> ids(new_ids, new_ids + new_count);
    try {
        return map_inference_error(s->fill(*ctx, ids));
    } catch (const std::exception& e) {
        set_last_error(std::string("session fill failed: ") + e.what());
        return PHONO_MODEL_ERROR;
    }
}

phono_status phono_session_replace_context(phono_session* session, phono_context* context,
                                           const int32_t* context_ids, int32_t context_count) {
    auto* s = reinterpret_cast<InferenceSession*>(session);
    auto* ctx = reinterpret_cast<phono::context::Context*>(context);
    if (s == nullptr || ctx == nullptr || context_count < 0 ||
        (context_count > 0 && context_ids == nullptr)) {
        return PHONO_INVALID_ARGUMENT;
    }
    const std::vector<int32_t> ids(context_ids, context_ids + context_count);
    try {
        return map_inference_error(s->replace_context(*ctx, ids));
    } catch (const std::exception& e) {
        set_last_error(std::string("session replace_context failed: ") + e.what());
        return PHONO_MODEL_ERROR;
    }
}

phono_status phono_session_generate(phono_session* session, phono_context* context,
                                    const int32_t* pinyin_ids, int32_t pinyin_count,
                                    const int32_t* context_ids, int32_t context_count,
                                    phono_cancel_fn cancel, void* cancel_user_data,
                                    phono_generate_result* out_result) {
    auto* s = reinterpret_cast<InferenceSession*>(session);
    auto* ctx = reinterpret_cast<phono::context::Context*>(context);
    if (s == nullptr || ctx == nullptr || out_result == nullptr) {
        return PHONO_INVALID_ARGUMENT;
    }
    if (pinyin_count < 0 || (pinyin_count > 0 && pinyin_ids == nullptr) ||
        context_count < 0 || (context_count > 0 && context_ids == nullptr)) {
        return PHONO_INVALID_ARGUMENT;
    }

    const std::vector<int32_t> pinyin(pinyin_ids, pinyin_ids + pinyin_count);
    const std::vector<int32_t> full_context(context_count > 0
                                                ? std::vector<int32_t>(context_ids,
                                                                       context_ids + context_count)
                                                : std::vector<int32_t>());
    CancelBridge bridge{cancel, cancel_user_data};
    GenerateResult generated;
    try {
        generated = s->generate(*ctx, pinyin, full_context, &cancellation_bridge, &bridge);
    } catch (const std::exception& e) {
        set_last_error(std::string("session generate failed: ") + e.what());
        return PHONO_MODEL_ERROR;
    }

    out_result->status = map_inference_error(generated.error);
    out_result->current_seqlen = generated.current_seqlen;
    out_result->history_seqlen = generated.history_seqlen;
    out_result->beam_count = 0;
    out_result->beams = nullptr;
    if (generated.beams.empty()) {
        return out_result->status;
    }

    auto* beams = static_cast<phono_beam*>(
        std::malloc(generated.beams.size() * sizeof(phono_beam)));
    if (beams == nullptr) {
        return PHONO_MODEL_ERROR;
    }
    out_result->beam_count = static_cast<int32_t>(generated.beams.size());
    out_result->beams = beams;
    for (size_t i = 0; i < generated.beams.size(); ++i) {
        const auto& beam = generated.beams[i];
        beams[i].score = beam.score;
        beams[i].pred_count = static_cast<int32_t>(beam.pred_ids.size());
        beams[i].pred_ids = nullptr;
        if (beam.pred_ids.size() > 0) {
            auto* pred = static_cast<int32_t*>(
                std::malloc(beam.pred_ids.size() * sizeof(int32_t)));
            if (pred == nullptr) {
                return PHONO_MODEL_ERROR;
            }
            std::memcpy(pred, beam.pred_ids.data(), beam.pred_ids.size() * sizeof(int32_t));
            beams[i].pred_ids = pred;
        }
        beams[i].decoded = nullptr;
        if (!beam.decoded.empty()) {
            auto* text = static_cast<char*>(std::malloc(beam.decoded.size() + 1));
            if (text == nullptr) {
                return PHONO_MODEL_ERROR;
            }
            std::memcpy(text, beam.decoded.c_str(), beam.decoded.size() + 1);
            beams[i].decoded = text;
        }
    }
    return PHONO_OK;
}

// ── misc ──────────────────────────────────────────────────────────────────

void phono_generate_result_free(phono_generate_result* result) {
    if (result == nullptr) {
        return;
    }
    if (result->beams != nullptr) {
        for (int32_t i = 0; i < result->beam_count; ++i) {
            std::free(result->beams[i].pred_ids);
            std::free(result->beams[i].decoded);
        }
        std::free(result->beams);
        result->beams = nullptr;
    }
    result->beam_count = 0;
}

void phono_free(void* ptr) { std::free(ptr); }

const char* phono_error_name(phono_status status) {
    switch (status) {
        case PHONO_OK: return "ok";
        case PHONO_INVALID_ARGUMENT: return "invalid_argument";
        case PHONO_CONTEXT_LIMIT_EXCEEDED: return "context_limit_exceeded";
        case PHONO_PINYIN_LIMIT_EXCEEDED: return "pinyin_limit_exceeded";
        case PHONO_INVALID_PINYIN: return "invalid_pinyin";
        case PHONO_NO_CANDIDATES: return "no_candidates";
        case PHONO_CANCELLED: return "cancelled";
        case PHONO_MODEL_ERROR: return "model_error";
        case PHONO_CONFIG_ERROR: return "config_error";
    }
    return "unknown";
}

const char* phono_last_error_message(void) { return g_last_error.c_str(); }

}  // extern "C"
