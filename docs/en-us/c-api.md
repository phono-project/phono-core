# C ABI call convention

`libphono_core` exports the exception-safe C ABI declared in
`interface/phono_api.h`. Handles are opaque and every failure is a
`phono_status`; `phono_last_error_message()` supplies a thread-local diagnostic.

## Typical flow

1. Load versioned `engine_config.json` with `phono_engine_create`.
2. Inspect segmenter availability and load diagnostics with
   `phono_engine_info_json`.
3. Send raw pinyin through `phono_engine_segment_pinyin`; its response already
   contains legal `pinyin_ids`.
4. Create a context manager and session using versioned
   `context_manager.json`, then call `phono_session_generate`.
5. Commit a candidate with `phono_session_fill` and repeat.

The segmentation endpoint automatically uses scorer-Viterbi when the package
contains a usable segmenter and checked FMM otherwise. Its complete request and
response schema is specified in [Smart Pinyin Segmentation](pinyin-segmentation.md).

```c
phono_engine* engine = NULL;
phono_status status = phono_engine_create(model_dir, engine_config_json, &engine);
if (status != PHONO_OK) {
    fprintf(stderr, "%s: %s\n", phono_error_name(status),
            phono_last_error_message());
}

char* result_json = NULL;
status = phono_engine_segment_pinyin(
    engine, "{\"schema_version\":\"1.0\",\"input\":\"nihaoma\"}",
    &result_json);
/* Parse pinyin_ids and invalid_ranges, then release the JSON. */
phono_free(result_json);
```

## Status codes

| Status | Meaning |
| --- | --- |
| `PHONO_OK` | Success |
| `PHONO_INVALID_ARGUMENT` | Null, negative count, invalid ID, or similar API misuse |
| `PHONO_CONTEXT_LIMIT_EXCEEDED` | Context exceeds the pre-model hard limit |
| `PHONO_PINYIN_LIMIT_EXCEEDED` | Character or syllable window exceeds its configured limit |
| `PHONO_INVALID_PINYIN` | No valid strict route, or safe deletion produced empty input |
| `PHONO_NO_CANDIDATES` | Generation found no valid candidate |
| `PHONO_CANCELLED` | Cancellation callback aborted generation and cursors were rolled back |
| `PHONO_MODEL_ERROR` | Package load or model execution failure |
| `PHONO_CONFIG_ERROR` | Invalid JSON configuration, request, or schema version |

`phono_error_name` returns the stable symbolic name.

## Configuration and sessions

Engine/tokenizer policy and context/session policy are different JSON inputs;
see [Runtime configuration](core-config.md). A session binds no context slot.
Every operation receives an explicit `phono_context`, so one session can drive
multiple slots. Passing no `context_ids` to generation reuses committed slot
history. A nonzero cancellation callback aborts between steps and restores
`current_seqlen` to `history_seqlen`.

## Ownership

- Strings returned by `phono_engine_info_json`,
  `phono_engine_segment_pinyin`, and `phono_tokenizer_decode` use `phono_free`.
- Arrays returned by `phono_tokenizer_encode_context` use `phono_free`.
- `phono_generate_result` internals use `phono_generate_result_free`.
- Engine, manager, and session handles use their matching destroy functions.
- Context handles are borrowed from the manager and must not be freed.

The complete declarations remain the authoritative function list in
`interface/phono_api.h`. `cli_demo_capi` and `ime_demo_capi` are end-to-end C
ABI examples; both display the selected segmentation route and invalid ranges.
