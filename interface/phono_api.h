// phono_api.h
//
// Stable C ABI for the phono-core inference engine. The library is exported
// as libphono_core.so / phono_core.dll / libphono_core.dylib; see the
// phono_core_shared CMake target.
//
// The runtime core_config is handed to the library as a plain JSON string
// (the same file that a demo or the package's core_configs/default.json
// carries). The library parses and validates it internally against the loaded
// model's hard limits and reports a phono_status error enum instead of
// crashing when a parameter does not fit.
//
// A session is stateless: no context is bound at creation and every call
// takes an explicit phono_context slot, so one session can drive many slots.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(PHONO_BUILDING_SHARED)
    #define PHONO_API __declspec(dllexport)
  #else
    #define PHONO_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define PHONO_API __attribute__((visibility("default")))
#else
  #define PHONO_API
#endif

typedef struct phono_engine phono_engine;
typedef struct phono_context_manager phono_context_manager;
typedef struct phono_context phono_context;
typedef struct phono_session phono_session;

typedef enum phono_status {
    PHONO_OK = 0,
    PHONO_INVALID_ARGUMENT,
    PHONO_CONTEXT_LIMIT_EXCEEDED,
    PHONO_PINYIN_LIMIT_EXCEEDED,
    PHONO_INVALID_PINYIN,
    PHONO_NO_CANDIDATES,
    PHONO_CANCELLED,
    PHONO_MODEL_ERROR,
    PHONO_CONFIG_ERROR,
} phono_status;

// Cooperative cancellation callback invoked between generation steps. Return
// nonzero to abort. user_data is an opaque pointer passed through unchanged.
typedef int (*phono_cancel_fn)(void* user_data);

typedef struct phono_beam {
    double score;
    int32_t* pred_ids;   // owned; free with phono_generate_result_free
    int32_t pred_count;
    char* decoded;       // owned UTF-8; free with phono_generate_result_free
} phono_beam;

typedef struct phono_generate_result {
    phono_status status;
    phono_beam* beams;   // owned; free with phono_generate_result_free
    int32_t beam_count;
    int32_t current_seqlen;
    int32_t history_seqlen;
} phono_generate_result;

// ------ engine ------
// Loads a model package directory with a versioned engine_config JSON object.
// Returns PHONO_MODEL_ERROR for package failures or PHONO_CONFIG_ERROR for an
// invalid runtime configuration.
PHONO_API phono_status phono_engine_create(const char* model_package_dir,
                                           const char* engine_config_json,
                                           phono_engine** out_engine);
PHONO_API void phono_engine_destroy(phono_engine* engine);
PHONO_API int32_t phono_engine_pre_max_seqlen(const phono_engine* engine);
PHONO_API int32_t phono_engine_post_max_seqlen(const phono_engine* engine);
PHONO_API int32_t phono_engine_beam_size(const phono_engine* engine);
// Returns a malloc-owned versioned JSON document; free with phono_free.
PHONO_API char* phono_engine_info_json(const phono_engine* engine);

// Versioned JSON request/response endpoint for automatic pinyin segmentation.
// A result document is returned even for PHONO_INVALID_PINYIN when allocation
// succeeds, so callers can display invalid_ranges.
PHONO_API phono_status phono_engine_segment_pinyin(const phono_engine* engine,
                                                   const char* request_json,
                                                   char** out_result_json);

// ------ tokenizer ------
// Output arrays are malloc-allocated and owned by the caller (free with
// phono_free). Decoded strings are malloc-allocated too.
// The separated syllable strings and their pointer array occupy one allocation;
// call phono_free once on the returned array, not on individual strings.
PHONO_API phono_status phono_tokenizer_encode_context(const phono_engine* engine,
                                                      const char* text_utf8,
                                                      int32_t** out_ids,
                                                      int32_t* out_count);
// Exact lookup only. Returns -1 when the token is absent.
PHONO_API int32_t phono_tokenizer_find_pinyin_id_exact(const phono_engine* engine,
                                                       const char* token);
PHONO_API char* phono_tokenizer_decode(const phono_engine* engine,
                                       const int32_t* ids, int32_t count);
PHONO_API int32_t phono_tokenizer_chinese_to_context(const phono_engine* engine,
                                                     int32_t chinese_id);

// ------ context manager ------
// core_config_json is the runtime core_config as a JSON string; it is parsed
// and validated against the loaded model. PHONO_CONFIG_ERROR is returned when
// any parameter exceeds the model limits.
PHONO_API phono_status phono_context_manager_create(phono_engine* engine,
                                                    const char* core_config_json,
                                                    int32_t num_contexts,
                                                    phono_context_manager** out_manager);
PHONO_API void phono_context_manager_destroy(phono_context_manager* manager);
PHONO_API int32_t phono_context_manager_num_contexts(const phono_context_manager* manager);
PHONO_API phono_context* phono_context_manager_get_by_id(phono_context_manager* manager,
                                                         int32_t id);
PHONO_API phono_context* phono_context_manager_get_auto(phono_context_manager* manager,
                                                        const int32_t* context_ids,
                                                        int32_t context_count);

// context accessors
PHONO_API int32_t phono_context_ids_len(const phono_context* context);
PHONO_API int32_t phono_context_current_seqlen(const phono_context* context);
PHONO_API int32_t phono_context_history_seqlen(const phono_context* context);

// ------ stateless session api ------
// core_config_json is parsed and validated as in phono_context_manager_create.
PHONO_API phono_status phono_session_create(phono_engine* engine,
                                            const char* core_config_json,
                                            phono_session** out_session);
PHONO_API void phono_session_destroy(phono_session* session);
PHONO_API int32_t phono_session_beam_size(const phono_session* session);
PHONO_API phono_status phono_session_reset(phono_session* session, phono_context* context);
PHONO_API phono_status phono_session_fill(phono_session* session, phono_context* context,
                                          const int32_t* new_ids, int32_t new_count);
PHONO_API phono_status phono_session_replace_context(phono_session* session,
                                                     phono_context* context,
                                                     const int32_t* context_ids,
                                                     int32_t context_count);
// Generates candidates. Pass context_count == 0 / NULL context_ids to reuse
// the committed history of `context`. On cancellation the context cursors are
// rolled back to the committed history.
PHONO_API phono_status phono_session_generate(phono_session* session, phono_context* context,
                                              const int32_t* pinyin_ids, int32_t pinyin_count,
                                              const int32_t* context_ids, int32_t context_count,
                                              phono_cancel_fn cancel, void* cancel_user_data,
                                              phono_generate_result* out_result);

// ------ misc ------
PHONO_API void phono_generate_result_free(phono_generate_result* result);
PHONO_API void phono_free(void* ptr);
PHONO_API const char* phono_error_name(phono_status status);
// Thread-local message describing the last failed phono_* call, if any.
PHONO_API const char* phono_last_error_message(void);

#ifdef __cplusplus
}  // extern "C"
#endif
