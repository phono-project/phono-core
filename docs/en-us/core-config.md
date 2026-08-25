# core_config Reference

The core_config is the runtime configuration, kept separate from the model package. It is a JSON object: the demos read it from the package's core_configs/default.json or take a custom JSON file as a command-line argument, and the C call convention hands the whole core_config to the shared library as a JSON string that the library parses and validates internally.

The configuration is validated against the model package's hard limits at parse time. When a parameter does not satisfy the constraints, an error enum code is returned (CoreConfigError on the C++ side, PHONO_CONFIG_ERROR on the C side) instead of throwing or silently falling back.

## Options

| Key | Description | Constraint | Default (model-derived) |
| --- | --- | --- | --- |
| beam_size | Beam-search width; must equal the model's beam/batch width | equal to the pre pass 2 batch size | model batch size |
| slack_interval | Window slack left over when truncation is triggered | 0 <= slack_interval < max_history_length | 8 |
| min_accept_context | Minimum reusable suffix length required for slot reuse | > 0 | 8 |
| max_context_length | Committed context-id capacity per slot | <= pre_model.max_seqlen - 1 | pre_model.max_seqlen - 1 |
| max_history_length | Soft cap on committed history; truncated before generation once reached | must satisfy max_history_length < max_context_length - max_pinyin_length | max_context_length - max_pinyin_length - 1 |
| max_pinyin_length | Upper bound of the pinyin window | >= 1 and <= post_model.max_seqlen (the post hard limit) | post_model.max_seqlen |
| trial_ratio | Fraction of slots allowed in the trial pool for eviction | 0 < trial_ratio <= 1 | 0.25 |
| decay_alpha | Exponent of slot value growth with length | >= 0 | 0.5 |
| decay_lambda | Rate of slot value decay over time | >= 0 | 1/60 |

Keys that are absent use the model-derived defaults, so even an empty object parses to a legal configuration for the current model. The legacy key names N and T have been replaced by slack_interval and min_accept_context and no longer have any effect.

## Model-Derived Defaults

Given the dimensions in the package's config.json, the defaults are derived as follows:

- beam_size is runtime.batch_size (the pre pass 2 batch size);
- max_context_length is pre_model.max_seqlen - 1;
- max_pinyin_length is post_model.max_seqlen;
- max_history_length is max_context_length - max_pinyin_length - 1, guaranteeing it stays strictly below max_context_length - max_pinyin_length;
- slack_interval is min(8, max_history_length - 1).

The shipped core_configs/default.json targets the v2_0_alpha_05 model (pre max_seqlen 128, post max_seqlen 32, batch 3): beam_size 3, slack_interval 8, min_accept_context 8, max_context_length 127, max_history_length 94, max_pinyin_length 32.

## Windowing Mechanism

The pre model has a hard sequence-length cap pre_model.max_seqlen and the post model has a hard input-length cap post_model.max_seqlen. To keep the context from growing unboundedly and the state from becoming inconsistent, two runtime caps govern the window:

- Pinyin window: the number of pinyin syllables fed to generate must not exceed max_pinyin_length, nor the post model's hard limit post_model.max_seqlen. PinyinLimitExceeded is returned otherwise.
- History window: the committed history must not exceed the soft cap max_history_length. When fill would push the history over the cap, it first truncates to max_history_length minus slack_interval (keeping the first BOS as the attention sink) and then appends; when generate starts and the history has already reached max_history_length, it truncates with the same rule first.

Truncation is a left-shift window: the oldest tokens are dropped and the cache is shifted token-wise, with BOS always occupying position zero. This keeps the context bounded while retaining as much recent context as possible.

## Consistency Constraints

The following constraints are enforced at parse time, each returning its own error:

- beam_size must equal the model's beam/batch width, otherwise BeamSizeMismatch;
- max_context_length + 1 must not exceed pre_model.max_seqlen, otherwise MaxContextLengthExceeded;
- max_pinyin_length must not exceed post_model.max_seqlen, otherwise MaxPinyinLengthInvalid;
- max_history_length must be strictly less than max_context_length - max_pinyin_length, otherwise MaxHistoryLengthInvalid. This guarantees that even when both the history and the pinyin window hit their caps, the committed text cannot overflow the pre cache before the next screen-up;
- slack_interval must lie in [0, max_history_length), otherwise SlackInvalid;
- min_accept_context must be greater than 0, otherwise MinAcceptContextInvalid.

The C++ error enum is core::CoreConfigError, convertible to a readable name with core_config_error_name; the C layer maps all of them to PHONO_CONFIG_ERROR with the reason available through phono_last_error_message.
