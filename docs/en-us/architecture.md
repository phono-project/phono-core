# Architecture

## Overview

The inference engine is composed of two models:

- The pre model processes the Chinese context and maintains a B-wide self-attention KV cache. It exports two methods: pre_model_pass1 incrementally pre-fills strictly causal history, and pre_model_pass2 generates candidate characters on top of the filled history. The cache tensor has shape [layers, 2, B, pre_max_seqlen, heads, head_dim].
- The post model encodes pinyin into hidden states and a bounded candidate-ID table. The runtime removes padding, evaluates lm_head only for real candidates, and maps sparse columns back to Chinese-vocabulary IDs. Input length is bounded by post_model.max_seqlen in config.json.

## Stateless Session and Context Slots

InferenceSession is stateless: the session stores no conversation state, and all state lives in the context slot that the caller passes explicitly. The session never binds a context at construction time; every context-dependent method (fill, replace_context, generate) takes the context as a parameter. This has two benefits: a single session can drive any number of context slots, which simplifies multi-session reuse and switching, and the C call convention can be designed around opaque handles plus explicit slots without coupling context lifetimes into the session.

Each context slot holds three things:

- a view of the pre-model self-attention KV cache (B-wide), pointing into the ContextManager's contiguous buffers;
- the committed text's context-vocabulary id sequence (BOS excluded);
- two cursors: history_seqlen is the committed strictly-causal history length (BOS included), and current_seqlen is the next writable cache position.

fill and replace_context modify both history_seqlen and current_seqlen; generate only advances current_seqlen and never changes history_seqlen. The cache positions written during generation are temporary, and the finally committed candidate is solidified into the history by fill.

## Incremental Fill

fill incrementally pre-fills newly committed strictly-causal history: when the incoming ids share a common prefix with the current history, only the new range runs pre_model_pass1 and beam zero's new slice is copied to the other beams. When the history would exceed the soft cap max_history_length, a left-shift window is applied: the oldest history tokens are dropped and the committed history is truncated to max_history_length minus slack_interval, always keeping the first BOS as the attention sink, then the new content is appended. replace_context handles three cases: an identical snapshot is reused directly; when history is a common prefix of the target, an incremental fill is performed; and divergent input is rebuilt from BOS.

## Generation and the Context Window

The generate flow is:

1. if the caller supplies a full context, replace_context first solidifies the target context into the slot;
2. if the history length has reached max_history_length, truncate it to max_history_length minus slack_interval, keeping BOS;
3. validate that the pinyin window does not exceed max_pinyin_length and that the generation positions stay below the pre model's hard cap;
4. run the post model to encode the pinyin, then perform B-way beam search token by token, reading the strictly causal history and the post-model cross-attention at each step;
5. return each beam's score, candidate ids and decoded text.

The values of max_history_length and max_pinyin_length must satisfy max_history_length < max_context_length - max_pinyin_length, otherwise the committed text could overflow before reaching the pre-model cache limit. The constraint is enforced at core_config parse time and reported as an error enum when violated.

## Cancellation and Cursor Rollback

generate checks the abort state through a callback-plus-context-pointer: the caller passes a boolean-returning callback and an opaque context pointer, and generate polls the callback between steps. When the callback returns true, generate returns the Cancelled error and rolls back the generation cursor, restoring current_seqlen to history_seqlen so the context returns to the consistent "committed history only" state. Interruption does not rewind the whole KV cache; the temporary state left in the generation positions is overwritten by the next generate or fill. This design lets an interrupted generate resume from the committed history without clearing the cache.

## Context Slot Management

ContextManager maintains a set of context slots, each holding contiguous-memory views of the pre-model self-attention KV cache and the context id sequence. Slots are evicted and reused by cost evaluation: prefix reuse truncates the trailing tail directly; BOS-protected left-shift reuse drops the oldest tokens and retains the matched suffix; when no reusable match exists, the weakest slot is chosen for recalculation. The minimum matched-suffix length is controlled by min_accept_context, the slack left for window reuse by slack_interval, and the slot value decays with usage frequency, controlled by trial_ratio, decay_alpha and decay_lambda.
