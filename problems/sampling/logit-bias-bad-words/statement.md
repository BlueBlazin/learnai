Implement three request-level controls that an OpenAI-compatible server applies to the next-token logits of one sequence before sampling. Your function applies an additive logit bias, enforces an allow-list of token ids, and prevents banned multi-token sequences from being completed. Banning a sequence removes only its final token, and only when the preceding tokens already appear at the very end of the sequence history.

## Specification

```python
def apply_bias(logits, bias, allowed_token_ids, bad_words, history): ...
```

The inputs represent one decoding step over a vocabulary of size $V$:

- `logits`: next-token logits for one sequence. Entries may already be `-inf`.
- `bias`: a dict mapping token ids to float offsets. It may be empty.
- `allowed_token_ids`: `None`, or a non-empty list of allowed token ids. Duplicates are possible.
- `bad_words`: a list of banned sequences, where each sequence is a non-empty list of token ids. The list may be empty. Sequences may overlap, repeat tokens, or share their final token.
- `history`: the list of token ids produced so far (prompt and generated tokens), oldest first. It may be empty.

The function returns the modified next-token logits.

A token $v \in \{0, \ldots, V - 1\}$ is **removed** if at least one of these two conditions holds:

1. **Allow-list.** `allowed_token_ids` is not `None` and $v$ is not in `allowed_token_ids`.
2. **Banned sequence.** Some sequence $w = (w_0, \ldots, w_{L-1})$ of length $L$ in `bad_words` ends with $w_{L-1} = v$, and `history` ends with its prefix $(w_0, \ldots, w_{L-2})$. Writing $h$ for `history` and $n = \text{len}(h)$, this prefix condition is

$$
L - 1 \le n \quad\text{and}\quad h_{n - L + 1 + j} = w_j \quad \text{for } j = 0, \ldots, L - 2.
$$

When $L = 1$, the prefix is empty and $L - 1 = 0 \le n$ always holds. A one-token banned sequence therefore removes its token at every step, even when `history` is empty. A sequence with $L - 1 > n$ cannot match.

Using the removal status and `bias`, each output entry is

$$
\text{out}[v] =
\begin{cases}
-\infty & \text{if } v \text{ is removed,} \\
\text{logits}[v] + \text{bias}[v] & \text{if } v \text{ is a key of } \texttt{bias}, \\
\text{logits}[v] & \text{otherwise.}
\end{cases}
$$

Key rules:

- **Removal wins over bias.** A removed token becomes `-inf` even if it has a `+100` bias.
- **Only the last token of a banned sequence is removed.** Earlier tokens in a banned sequence are never removed, and the last token is removed only when the earlier tokens appear in order at the very end of `history`. Matching tokens earlier in `history` or in a different order do not trigger the ban.
- **Sequences are checked independently.** Multiple banned sequences can match at the same step and remove different tokens.
- **The allow-list does not protect a token.** An allowed token that completes a banned sequence is still removed.
- **Bias is additive.** A logit that is already `-inf` stays `-inf`.
- **No renormalization.** The function returns raw logits and does not check whether any finite logit remains.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(V,)` | float32 |
| result | `(V,)` | float32 |

## Example

```python
logits = [1.0, 2.0, 0.5, -1.0, 3.0, 0.0, 1.5, -0.5]
bias = {1: -1.5, 6: 2.0, 7: 100.0}
allowed_token_ids = [0, 1, 2, 3, 4, 6]
bad_words = [[4], [5, 2, 3], [2, 6], [2, 5, 0], [3, 2, 1]]
history = [3, 5, 2]

apply_bias(logits, bias, allowed_token_ids, bad_words, history)
# [1.0, 0.5, 0.5, -inf, -inf, -inf, -inf, -inf]
```

| token | result | reason |
|---|---|---|
| 0 | `1.0` | Kept. `[2, 5, 0]` does not match because `history` ends with `5, 2`, not `2, 5`. |
| 1 | `0.5` | Kept with bias $2.0 - 1.5 = 0.5$. `[3, 2, 1]` does not match because `history` ends with `5, 2`, not `3, 2`. |
| 2 | `0.5` | Kept. Token 2 appears inside banned sequences, never as the final token. |
| 3 | `-inf` | Removed because `[5, 2, 3]` matches the tail `5, 2` of `history`. |
| 4 | `-inf` | Removed because `[4]` is a one-token banned sequence. |
| 5 | `-inf` | Removed because it is not in `allowed_token_ids`. |
| 6 | `-inf` | Removed because `[2, 6]` matches the tail `2` of `history`. Neither its `+2.0` bias nor the allow-list saves it. |
| 7 | `-inf` | Removed because it is not in `allowed_token_ids`, regardless of its `+100.0` bias. |

## Constraints

- $1 \le V \le 5000$.
- All token ids are integers in $[0, V)$.
- Bias values lie in $[-100, 100]$, and finite logits lie in $[-1000, 1000]$.
- There are at most 64 banned sequences, each with at most 8 tokens.
- `history` has at most 2000 tokens.
- Kept entries are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. Removed entries must be exactly `-inf`.
- Python loops over `bias`, `allowed_token_ids`, and `bad_words` are expected. No loop over the vocabulary is needed.

## Why this matters

These controls correspond to `logit_bias`, `allowed_token_ids`, and `bad_words` in vLLM's completion API (and `bad_words_ids` / `sequence_bias` in Hugging Face `transformers`). They run on every decoding step of every request that uses them, so small logic bugs directly break user-facing guarantees.

Because words span multiple tokens, banning a word means banning a token sequence. Removing every token in the word deletes common subwords from the vocabulary and degrades unrelated output, while matching a prefix anywhere in the history keeps the final token banned long after the context has moved on. Similarly, classification and tool-choice endpoints rely on allow-lists to restrict output to valid labels. If a `+100` bias overrides the mask, a guaranteed-valid endpoint can emit an invalid token.
