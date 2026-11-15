Implement the three request-level controls that an OpenAI-compatible server applies to
the logits of one sequence before sampling: an additive logit bias, an allow-list of
token ids, and banned multi-token sequences. Each is a few lines. The exercise is in
how they combine, and in banning a sequence without banning its tokens: only the
token that would *complete* a banned sequence is removed, and only at the step where
the text generated so far ends with the rest of it.

## Specification

```python
def apply_bias(logits, bias, allowed_token_ids, bad_words, history): ...
```

- `logits`: a float32 tensor of shape `(V,)`, the next-token logits of one sequence.
  Entries may already be `-inf`.
- `bias`: a dict from token id to a float. May be empty.
- `allowed_token_ids`: `None`, or a non-empty list of token ids (duplicates are
  possible).
- `bad_words`: a list of banned sequences, each a non-empty list of token ids. May be
  empty; sequences may overlap, repeat tokens, and share their last token.
- `history`: the list of token ids of the sequence so far (prompt and generated
  tokens), oldest first. May be empty.
- Returns a float32 tensor of shape `(V,)`.

A token $v$ is **removed** if at least one of these holds:

1. **Allow-list.** `allowed_token_ids` is not `None` and $v$ is not in it.
2. **Banned sequence.** Some sequence $w = (w_1, \ldots, w_L)$ in `bad_words` has
   $w_L = v$, and `history` ends with $w_1, \ldots, w_{L-1}$: with
   $h = $ `history` and $n = $ `len(history)`,

$$
L - 1 \le n \quad\text{and}\quad h_{n-L+1+j} = w_{1+j} \ \text{ for } j = 0, \ldots, L-2 .
$$

   For $L = 1$ the condition is empty: a one-token sequence removes its token at every
   step, whatever the history, also when the history is empty. A sequence with
   $L - 1 > n$ cannot match.

The result is

$$
\text{out}[v] =
\begin{cases}
-\infty & \text{if } v \text{ is removed,} \\
\text{logits}[v] + \text{bias}[v] & \text{if } v \text{ is a key of } \texttt{bias}, \\
\text{logits}[v] & \text{otherwise.}
\end{cases}
$$

Points that follow, and that the tests check:

- **Removal wins over bias.** A removed token is `-inf` even if its bias is `+100`.
- **Only the last token of a banned sequence is removed**, never its earlier tokens,
  and only while the history ends with the earlier tokens *in order and at the very
  end*. The same tokens earlier in the history, or in another order, do not count.
- Every banned sequence is checked independently; several can match at one step and
  remove different tokens.
- The allow-list does not protect a token: an allowed token that completes a banned
  sequence is removed.
- The bias is added, not assigned, and a logit that is already `-inf` stays `-inf`.
- Nothing is renormalized, and the function does not check that a finite logit
  remains.

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

| token | why |
|---|---|
| 0 | kept. `[2, 5, 0]` does not match: the history ends with `5, 2`, not `2, 5`. |
| 1 | kept, with bias: $2.0 - 1.5 = 0.5$. `[3, 2, 1]` does not match: the history contains 3 and 2, but ends with `5, 2`. |
| 2 | kept. Token 2 occurs inside banned sequences, never as their last token. |
| 3 | removed: `[5, 2, 3]` and the history ends with `5, 2`. |
| 4 | removed: `[4]` is a one-token sequence. |
| 5 | removed: not in the allow-list. |
| 6 | removed: `[2, 6]` and the history ends with `2`. Its bias of `+2` and its place in the allow-list do not matter. |
| 7 | removed: not in the allow-list, whatever its bias. |

## Constraints

- $1 \le V \le 5000$. Token ids everywhere are ints in $[0, V)$. Bias values lie in
  $[-100, 100]$; finite logits lie in $[-1000, 1000]$.
- At most 64 banned sequences of at most 8 tokens each; `history` has at most 2000
  tokens.
- The result is compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`; removed entries must be exactly `-inf`.
- Python loops over `bias`, `allowed_token_ids`, and `bad_words` are expected. No loop
  over the vocabulary is needed.

## Why this matters

These are the parameters `logit_bias`, `allowed_token_ids`, and `bad_words` of vLLM's
completion API (the first is OpenAI's; Hugging Face `transformers` has `bad_words_ids`
and `sequence_bias`). They run on every step of every request that uses them, and
they are where a serving bug becomes a product bug. Words are several tokens long, so
"never say X" is a statement about sequences: banning every token of the word removes
common word pieces from the model's vocabulary and degrades unrelated text, while
matching the prefix anywhere in the history bans the last token long after the
context has moved on. And an allow-list is how classification and tool-choice
endpoints restrict the output to a few labels: a `+100` bias that revives a token
outside it turns a guaranteed-valid response into an invalid one.
