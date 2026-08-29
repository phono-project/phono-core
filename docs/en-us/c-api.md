# C-ABI Call Convention

The shared library libphono_core.so (phono_core.dll on Windows, libphono_core.dylib on macOS) exports a stable C call convention declared in interface/phono_api.h. The interface uses opaque handles, no C++ exception ever crosses the ABI boundary, and every failure is reported as an error enum code.

## Call Flow

A typical call sequence is:

1. phono_engine_create loads the model package directory and returns an engine handle;
2. phono_context_manager_create takes the core_config JSON string and the number of slots, returning a context-manager handle;
3. phono_context_manager_get_auto or phono_context_manager_get_by_id returns a context-slot handle;
4. phono_session_create takes the same core_config JSON string and returns a session handle (the session does not bind a slot);
5. loop: phono_tokenizer_separate_greedy segments raw pinyin, phono_tokenizer_encode_pinyin encodes it, phono_session_generate produces candidates, phono_generate_result_free releases the result, and phono_session_fill commits the chosen candidate;
6. release the session, the manager and the engine in order when done.

## Status Codes

| Enum | Meaning |
| --- | --- |
| PHONO_OK | success |
| PHONO_INVALID_ARGUMENT | invalid argument (null pointer, negative length, out-of-range id, ...) |
| PHONO_CONTEXT_LIMIT_EXCEEDED | context exceeds the pre model's hard cap |
| PHONO_PINYIN_LIMIT_EXCEEDED | pinyin window exceeds max_pinyin_length or the post hard limit |
| PHONO_NO_CANDIDATES | not enough candidates to fill the beam |
| PHONO_CANCELLED | the callback requested an abort; cursors were rolled back |
| PHONO_MODEL_ERROR | model load or execution failure |
| PHONO_CONFIG_ERROR | core_config failed to parse or a parameter exceeds a model limit |

phono_error_name returns a stable name for a status code; phono_last_error_message returns the thread-local diagnostic message of the most recent failure.

## Passing the core_config

The context-manager and session create functions accept the core_config as a JSON string, for example:

```c
const char* config_json =
    "{\"beam_size\":3,\"slack_interval\":8,\"min_accept_context\":8,"
    "\"max_context_length\":127,\"max_history_length\":94,\"max_pinyin_length\":32}";
phono_context_manager* manager = NULL;
if (phono_context_manager_create(engine, config_json, 1, &manager) != PHONO_OK) {
    fprintf(stderr, "%s\n", phono_last_error_message());
}
```

The library deserializes it internally with nlohmann-json and validates it against the model limits; when a parameter does not fit (for example the beam or the context length exceeds a model limit) it returns PHONO_CONFIG_ERROR. Callers do not need to reconstruct any model-internal structure — just hand over the JSON string.

## Sessions and Slots

A session is stateless: it binds no slot at creation and every method takes an explicit phono_context handle, so a single session can drive any slot. The generate pinyin window is capped by core_config max_pinyin_length and the context by max_history_length; an empty context_ids means "reuse the history already committed in the slot". Aborts are handled through a callback polled between steps; a nonzero return aborts and rolls the cursors back:

```c
int my_cancel(void* user_data) {
    (void)user_data;
    return g_stop_flag;   /* set by a signal or another thread */
}
```

## Memory Ownership

- The id arrays returned by the encoding functions (phono_tokenizer_encode_context, phono_tokenizer_encode_pinyin) are owned by the caller and released with phono_free;
- the syllable array returned by phono_tokenizer_separate_greedy is one contiguous allocation and is released with one phono_free call;
- the string returned by phono_tokenizer_decode is released with phono_free;
- the internals of phono_generate_result (beams, pred_ids, decoded) filled by phono_session_generate are released together by phono_generate_result_free;
- the handles are released by their matching destroy functions: phono_engine_destroy, phono_context_manager_destroy and phono_session_destroy.

## Full Function List

See the declarations and comments in interface/phono_api.h, including phono_engine_create/destroy, phono_engine_pre_max_seqlen, phono_engine_post_max_seqlen, phono_engine_beam_size, phono_tokenizer_separate_greedy, phono_tokenizer_encode_context, phono_tokenizer_encode_pinyin, phono_tokenizer_decode, phono_tokenizer_chinese_to_context, phono_context_manager_create/destroy/num_contexts/get_by_id/get_auto, phono_context_ids_len, phono_context_current_seqlen, phono_context_history_seqlen, phono_session_create/destroy/beam_size/reset/fill/replace_context/generate, phono_generate_result_free, phono_free, phono_error_name and phono_last_error_message.

## Demo

streaming_benchmark_demo_capi is a complete example of using this C call convention: it segments a raw pinyin line, generates candidates and commits the chosen one. Its source is in apps/streaming_benchmark_demo_capi.cpp.
