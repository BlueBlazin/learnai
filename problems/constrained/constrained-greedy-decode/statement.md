Implement greedy decoding with a KV cache under a token constraint. At each step, choose only among the tokens permitted by the constraint, and allow end-of-sequence (`eos_id`) only when the output so far is complete. Keep the constraint state synchronized with the generated tokens while passing each token through the model at most once.

![This problem is the decode loop, which masks, picks, and advances the state once per token.](assets/map.svg)

## Specification

```python
def constrained_greedy(lm, prompt, constraint, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), used the same way as in `cached-greedy-decode`. Calling `logits, cache = lm(tokens)` starts a sequence, `logits, cache = lm(tokens, cache)` extends the cached prefix, and `logits[-1]` (shape `(vocab_size,)`, where `vocab_size` is the vocabulary size) gives the next-token logits.

`constraint` is a [`TokenConstraint`](../../../python/src/learnai/kit/constraint.py) that defines valid outputs through four methods:

```python
state = constraint.start()                 # initial state before any token is generated
constraint.allowed(state)                  # sorted list of token IDs that may come next
state = constraint.advance(state, token)   # next state after `token`; raises ValueError if not allowed
constraint.is_accepting(state)             # True if the output so far is complete
```

States are opaque, hashable values, and `advance` returns a new state without mutating the old one. `allowed(state)` may be empty and never contains `eos_id`. Whether the output may end is determined solely by `is_accepting(state)`. The constraint applies only to the generated tokens, not to `prompt`.

### Inputs and return value

- `prompt`: a list of $P \ge 1$ token IDs (Python `int`).
- `max_new_tokens`: an integer $\ge 1$ capping the number of generated tokens.
- `eos_id`: an integer token ID for the vocabulary's end-of-sequence token.
- Returns the generated tokens (excluding `prompt`) as a list of $g$ Python `int`s, with $0 \le g \le$ `max_new_tokens`.

### One step

When the constraint is in state $s$, the set of *permitted* tokens is

$$
\text{permitted}(s) = \text{allowed}(s) \;\cup\; \begin{cases} \{\texttt{eos\_id}\} & \text{if is\_accepting}(s) \\ \emptyset & \text{otherwise.} \end{cases}
$$

Choose the permitted token with the largest next-token logit, breaking ties by the lowest token ID. Equivalently, set the logit of every non-permitted token to $-\infty$ and take `argmax`. Advance the constraint with the chosen token unless it is `eos_id`.

### The loop

Start at $s =$ `constraint.start()` and repeat until the first of these conditions holds:

1. **Nothing is permitted** in the current state (`permitted(s)` is empty). The output cannot continue and may not end. Return the tokens generated so far without `eos_id` (returning `[]` if this happens on the first step).
2. **`eos_id` is generated.** Append `eos_id` as the final element and stop.
3. **`max_new_tokens` tokens** have been generated. Stop and return them, even if the output is not yet complete.

### Cost limits

As in `cached-greedy-decode`, a call that returns $g$ tokens from a prompt of length $P$ must stay within these limits:

| counter | limit | why |
|---|---|---|
| `lm.counters.forward_calls` | $g$ | one call for the prompt (producing the first token), then one call per additional token |
| `lm.counters.tokens_processed` | $P + g - 1$ (or $0$ if $g = 0$) | each prompt token once, and each generated token once except the last |

This imposes two requirements. First, never feed the final generated token back into `lm`, regardless of which stopping rule triggered. Second, query the constraint before calling `lm` so that a dead end costs zero forward calls (and a dead end at step 1 does not call `lm` at all).

## Shapes

In the table below, $T$ is the number of tokens passed in a single call to `lm`.

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` | float32 |
| `constraint.allowed(state)` | sorted list of ints | Python `int` |
| result | list of $g$ ints | Python `int` |

## Example

Consider the regular expression `\[\d(, \d)*\]`, which matches a list of digits such as `[4]` or `[0, 7, 7]`, over the kit's 64-token vocabulary:

```python
lm = TinyLM(fw, seed=5)
vocab = ToyVocab(64, seed=1)
constraint = DFAConstraint(CharDFA.from_regex(r"\[\d(, \d)*\]"), vocab)

constrained_greedy(lm, [0, 17, 42, 5], constraint, 12, eos_id=1)
# [51, 54, 7, 32, 39, 45, 54, 61, 1]      text "[5, 1, 5]", then EOS
# 9 calls, 12 tokens processed
```

| step | text so far | permitted | unconstrained argmax | chosen |
|---|---|---|---|---|
| 1 | (empty) | `[` | 37 `e` | 51 `[` |
| 2 | `[` | the ten digits | 37 `e` | 54 `5` |
| 3 | `[5` | `,` or `, ` or `]` | 37 `e` | 7 `,` |
| 4 | `[5,` | the space | 2 (a special token) | 32 (space) |
| 5 | `[5, ` | the ten digits | 37 `e` | 39 `1` |
| 6 | `[5, 1` | `,` or `, ` or `]` | 33 `.` | 45 `, ` |
| 7 | `[5, 1, ` | the ten digits | 30 (half of an emoji) | 54 `5` |
| 8 | `[5, 1, 5` | `,` or `, ` or `]` | 34 `€` | 61 `]` |
| 9 | `[5, 1, 5]` | EOS | 11 `3` | 1 (EOS) |

Unconstrained, this model would never emit a valid list because its top logit is invalid at every step. Masking restricts each choice to the permitted tokens. Notice three details:

- **Multi-character tokens.** After a digit, the vocabulary contains both `,` (token 7) and the two-character piece `, ` (token 45). Step 3 picks `,`, leaving only the space permitted at step 4. Step 6 picks `, ` in a single token. The loop never inspects token text and relies solely on `allowed`.
- **EOS gating.** `eos_id` is excluded during steps 1 through 8 because the list is incomplete. At step 9, the state is accepting and no other token can extend the list, so `eos_id` is the only permitted choice. For a pattern like `-?\d+`, `eos_id` and the ten digits would both be permitted after every digit, and the logits would decide whether to stop or continue.
- **Forced-step accounting.** Step 9 still performs a forward call even though only `eos_id` is permitted. The loop makes 9 forward calls and processes $4 + 8 = 12$ tokens in total, never feeding token `1` back to `lm`.

## Constraints

- $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens` $\le$ `lm.max_context` (128).
- Generated tokens are compared exactly against uncached constrained greedy decoding. In all test cases, the top two permitted logits differ by more than `1e-3`, so results are independent of call batching and ties do not occur.
- Both cost counters are checked strictly against the limits above. A solution that exceeds either limit fails even if its tokens match.
- Use only the four documented methods of `constraint`. Tests include non-DFA constraints whose states are tuples.

## Why this matters

JSON mode, schema-guided function calling, and regex- or grammar-constrained generation all rely on this loop. The constraint engine (such as a DFA over the vocabulary or a pushdown automaton for JSON) decides which tokens are valid next, and the decode loop enforces that guarantee by masking logits before selection.

Small bugs in the loop break the guarantee silently. If the constraint state falls out of sync with the generated tokens, the mask applies to the wrong position. If `eos_id` is left unmasked, the model can stop mid-string and return broken JSON. If `eos_id` is never permitted, the model loops until `max_new_tokens`. Finally, validating tokens after the forward pass instead of before selection wastes a model call on every rejected candidate along the hot path of structured generation.
