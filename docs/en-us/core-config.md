# Runtime configuration reference

Runtime policy is deliberately separate from the model package and split into two versioned JSON documents. Every document requires the string `"schema_version": "1.0"`; missing and unknown versions return `PHONO_CONFIG_ERROR`.

## Engine and tokenizer policy

`core_configs/engine_config.json` is passed to `phono_engine_create`:

```json
{
  "schema_version": "1.0",
  "tokenizer": {
    "normalization": {
      "lowercase_ascii": true,
      "normalize_v_to_u": true,
      "separators": "'"
    },
    "segment_mode": "safe",
    "repair": false,
    "max_pinyin_chars": 128
  }
}
```

| Key | Meaning |
| --- | --- |
| `lowercase_ascii` | Convert ASCII uppercase letters before matching |
| `normalize_v_to_u` | Convert `v` to `u` only after `j/q/x/y` in the same separator-delimited run |
| `separators` | Characters removed while forcing a syllable boundary |
| `segment_mode` | `safe` returns valid output plus invalid ranges; `strict` rejects any invalid route |
| `repair` | In safe mode, repair each invalid character to its nearest legal token instead of deleting it |
| `max_pinyin_chars` | Maximum normalized character count accepted by the segmentation endpoint |

`max_pinyin_chars` limits raw segmentation work. It is distinct from `max_pinyin_length`, which limits the number of syllable IDs passed to the P2C model. See [Smart Pinyin Segmentation](pinyin-segmentation.md) for routing, normalization, and response schemas.

## Context and session policy

`core_configs/context_manager.json` is passed independently to `phono_context_manager_create` and `phono_session_create`:

```json
{
  "schema_version": "1.0",
  "beam_size": 3,
  "slack_interval": 8,
  "min_accept_context": 8,
  "max_context_length": 127,
  "max_history_length": 94,
  "max_pinyin_length": 32,
  "trial_ratio": 0.25,
  "decay_alpha": 0.5,
  "decay_lambda": 0.02
}
```

| Key | Constraint | Default when omitted |
| --- | --- | --- |
| `beam_size` | Equals the pre pass-2 batch size | Model batch size |
| `slack_interval` | `0 <= value < max_history_length` | Up to 8 |
| `min_accept_context` | Greater than zero | 8 |
| `max_context_length` | At most `pre_model.max_seqlen - 1` | That maximum |
| `max_history_length` | Less than `max_context_length - max_pinyin_length` | Largest legal value |
| `max_pinyin_length` | `1..post_model.max_seqlen` | Post-model maximum |
| `trial_ratio` | `(0,1]` | 0.25 |
| `decay_alpha` | Non-negative | 0.5 |
| `decay_lambda` | Non-negative | 1/60 |

Fields other than `schema_version` may be omitted and use model-derived defaults. Unknown legacy names such as `N` and `T` have no effect.

The history is a BOS-preserving left-shift window. When it reaches `max_history_length`, old tokens are discarded down to `max_history_length - slack_interval`. Violating a model limit returns `CoreConfigError` in C++ or `PHONO_CONFIG_ERROR` in C, with detail in `phono_last_error_message`.
