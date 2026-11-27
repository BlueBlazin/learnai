Implement top-k filtering for a batch of next-token logits. For each row, keep exactly the `k` highest-ranked logits and mask all remaining entries with `-inf`, breaking ties by token index and handling edge values of `k`.

![This problem is Top-k, which masks all but the `k` largest logits.](assets/map.svg)

## Specification

```python
def top_k_filter(logits, k): ...
```

The input `logits` holds `B` rows of next-token scores over a vocabulary of `V` tokens, and `k` is a non-negative Python `int` applied to every row. Each row is filtered independently.

Rank the tokens in row $b$ by decreasing logit value. **Break ties by token index, lowest first.** Keep the first `k` tokens in that ranking and set all others to `-inf`:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if token } v \text{ is among the first } k \text{ in the ranking of row } b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

Input and output value guarantees:

- `logits` may contain `-inf` for already-masked tokens, and every row has at least one finite entry. Neither `+inf` nor NaN appears.
- Every kept entry retains its exact input value, and every removed entry is exactly `-inf`.

Key rules and edge cases:

- **Exactly `k` positions are kept.** When the k-th largest value is tied across multiple tokens, only the tied tokens with the lowest indices are kept so the total stays `k`.
- **Masked entries stay masked.** A `-inf` entry ranks below every finite logit. If a row has fewer than `k` finite entries, all finite entries are kept and the rest remain `-inf`, leaving $\min(k, \text{number of finite entries in the input row})$ finite entries.
- **$k \ge V$ is a no-op.** Return `logits` unchanged.
- **$k = 0$ disables top-k filtering.** Return `logits` unchanged. This follows the convention of llama.cpp and most sampler configs rather than keeping zero tokens.

You may use your framework's sorting functions.

> [!WARNING]
> A built-in `topk` does not guarantee which tied entries it returns, and the default sort in some frameworks is not stable. The lowest-index tie-breaking rule above is part of the specification.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| result | `(B, V)` | float32 |

## Example

```python
logits = [[1.0, 3.0, 0.5, 2.0, -1.0], [0.0, 2.0, -inf, 2.0, 2.0]]

top_k_filter(logits, 2)
# [[-inf, 3.0, -inf, 2.0, -inf],
#  [-inf, 2.0, -inf, 2.0, -inf]]
```

Row 0 has no ties. Its two largest logits are `3.0` (token 1) and `2.0` (token 3).

Row 1 has three logits equal to `2.0` at tokens 1, 3, and 4, with only two slots available. Breaking ties by lowest index gives the ranking `1, 3, 4`, followed by token 0 (`0.0`) and the masked token 2 (`-inf`). Tokens 1 and 3 are kept, while token 4 is masked even though its logit matches the smallest kept value. Keeping every logit greater than or equal to the k-th largest value would wrongly keep three tokens here.

With `k = 4`, row 1 keeps tokens 1, 3, 4, and 0, while the masked token 2 stays `-inf`. With `k = 0`, `k = 5`, or `k = 100`, both rows are returned unchanged.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$, $0 \le k \le 10^6$. Finite logits lie in $[-10^5, 10^5]$.
- Because the function only compares and copies values, kept entries must match the input (relative tolerance `1e-4`, absolute tolerance `1e-6`), and removed entries must be exactly `-inf`.
- Ties occur in most tests, including ties at the k-th boundary and rows where all logits are equal.
- No Python loop over rows or tokens is needed.

## Why this matters

Top-k is usually the first filter in a sampler chain (for example, llama.cpp defaults to `k = 40`). It improves both quality and speed because later stages only process 40 candidates instead of a 150,000-token vocabulary.

Most top-k bugs happen at the boundaries. A value-threshold filter keeps every token tied with the k-th logit. That looks harmless on distinct float32 scores, but quantized models, masked rows, and padded vocabularies frequently produce many identical logits, sending more candidates downstream than later stages expect. Likewise, clients pass `k = 0` when top-k is unset, so treating `0` literally masks the entire vocabulary and leaves the sampler with nothing to pick.
