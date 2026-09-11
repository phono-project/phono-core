# Smart Pinyin Segmentation

## Scorer and MAP decoding

The segmentation model scores character gaps; the pinyin-vocabulary Trie enforces legality. A normalized input of length `n` always has `n-1` gaps. If the model emits logit `z_k` and boundary state `y_k`, then

```text
log P(Y|X) = sum_k [y_k log sigmoid(z_k) + (1-y_k) log(1-sigmoid(z_k))]
           = C(X) + sum_k y_k z_k.
```

`C(X)` is path-independent. For MAP argmax only, the decoder therefore sums raw logits at selected boundaries. No sigmoid, geometric mean, or syllable count penalty is needed, even when full pinyin and jianpin produce different numbers of syllables. This simplification preserves argmax; it is not a calibrated whole-path probability.

All full-pinyin and jianpin matches from each offset form a forward Trie DAG. An edge `(i,j)` receives `z_(j-1)` when `j<n`, or zero at the input end. The dynamic program is `O(E)`. Exact-score ties prefer fewer syllables, then longer earlier syllables.

## Automatic routing

`InferenceEngine::segment_pinyin` and C ABI `phono_engine_segment_pinyin` are the stable high-level entry points:

- a package segmenter and an input at least `min_input_chars` long use the scorer plus Trie DAG;
- lengths one and two bypass the model and use checked FMM;
- a package without `segmenter` remains usable, routes every input to checked FMM, and reports a warning through `phono_engine_info_json.diagnostics`;
- a declared but broken segmenter fails engine loading instead of silently falling back.

Checked FMM includes global reachability: it first minimizes invalid characters, then applies maximum-matching tie breaks. A locally longest token cannot create an avoidable dead end.

## Normalization and invalid input

Normalization is configured by `engine_config.json`: ASCII lowercasing; separator removal with a forced boundary; and `v -> u` after `j/q/x/y` when no forced boundary intervenes. Thus `jvan` becomes `juan`, while `lv` and `nv` stay unchanged. Every change is recorded in `normalization_events`.

Strict mode accepts only legal Trie edges. An unreachable input returns `InferenceError::InvalidPinyin` / `PHONO_INVALID_PINYIN` without running the scorer.

Safe mode retains per-character invalid edges in the DP but minimizes invalid characters before considering model score, so they are selected only when no fully legal route exists:

- `repair=false` (default) deletes selected invalid characters;
- `repair=true` repairs each invalid character independently via `find_pinyin_id_nearest`; edit distance is not exposed.

`invalid_ranges` uses half-open UTF-8 byte offsets `[begin,end)` in the original pre-normalization string. On safe success, `normalized_input` and every entry in `segments` are exactly encodable by the pinyin tokenizer. A result that is empty after deletion still returns `PHONO_INVALID_PINYIN`.

## Versioned JSON schema 1.0

Every C ABI JSON document requires top-level string `schema_version: "1.0"`. Missing or unknown versions return `PHONO_CONFIG_ERROR`.

Segmentation request:

```json
{"schema_version":"1.0","input":"jv'anpv"}
```

Segmentation response (also allocated on `PHONO_INVALID_PINYIN` when possible):

```json
{
  "schema_version": "1.0",
  "status": {"ok": true, "code": "ok"},
  "input": "jv'anpv",
  "canonical_input": "juanpv",
  "normalized_input": "juanp",
  "strategy": "scorer_viterbi",
  "segments": ["juan", "p"],
  "pinyin_ids": [167, 9],
  "invalid_ranges": [
    {"begin": 6, "end": 7, "text": "v", "replacement": ""}
  ],
  "normalization_events": [
    {"type":"v_to_u","begin":1,"end":2,"before":"v","after":"u"},
    {"type":"forced_boundary","begin":2,"end":3,"before":"'","after":""},
    {"type":"invalid_deleted","begin":6,"end":7,"before":"v","after":""}
  ]
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| status | object | Boolean `ok` plus stable error name `code` |
| canonical_input | string | Scorer input after character normalization and separator removal, before invalid handling |
| normalized_input | string | Guaranteed-legal final pinyin string on success |
| strategy | string | `scorer_viterbi` or `checked_fmm` |
| segments | string[] | Final legal tokens, one-to-one with `pinyin_ids` |
| pinyin_ids | int32[] | Ready for `phono_session_generate` |
| invalid_ranges | object[] | Original byte range, original text, and replacement; empty replacement means deletion |
| normalization_events | object[] | Every normalization, deletion, or repair in execution order |

`phono_engine_info_json` returns:

```json
{
  "schema_version":"1.0",
  "model_version":"v2_2-base-alpha05-w8a8",
  "model_format_version":"2.2",
  "segmenter":{"available":true,"fallback":null},
  "diagnostics":[]
}
```

The optional package `config.json` section is below. `layout` is fixed to `BHWC`, and `min_input_chars` is at least three:

```json
"segmenter": {
  "model_path": "bins/pinyin_segment.pte",
  "method": "forward",
  "char_vocab": "vocabs/pinyin_char_vocab.txt",
  "min_input_chars": 3,
  "max_input_chars": 512,
  "layout": "BHWC",
  "quantization": "none"
}
```
