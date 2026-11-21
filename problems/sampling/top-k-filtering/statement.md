Implement top-k filtering for a batch of logits: in every row, keep the `k` largest
logits and mask all the others with `-inf`. Finding the k-th largest value is easy;
the exercise is to keep **exactly** `k` entries when several logits are tied at the
boundary, and to get the edge values of `k` right.

![This problem is Top-k, which masks all but the `k` largest logits.](assets/map.svg)

## Specification

```python
def top_k_filter(logits, k): ...
```

Each row of `logits` is treated independently. Rank the tokens of a row by decreasing
logit. **Tokens with equal logits are ranked by token index, lowest first.** The `k`
highest-ranked tokens are kept:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if token } v \text{ is among the first } k \text{ in the ranking of row } b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (a masked
  token); every row has at least one finite entry. `+inf` and NaN never appear.
- `k`: a Python int, $k \ge 0$, the same for every row.
- Returns a float32 tensor of shape `(B, V)`. A kept entry has exactly its input
  value; a removed entry is exactly `-inf`.

Three consequences of the definition, and one convention:

- **Exactly `k` positions are kept**, also when the k-th largest value occurs several
  times. Of the tied entries, the ones with the lowest indices are kept.
- **Masked entries stay masked.** A `-inf` entry ranks below every finite one. If a row
  has fewer than `k` finite entries, all of them are kept and the rest of the row is
  `-inf` as before, so the number of finite entries in a row of the result is
  $\min(k, \text{number of finite entries in the input row})$.
- **$k \ge V$ changes nothing**: the logits are returned unchanged.
- **$k = 0$ means "top-k is disabled"** and also returns the logits unchanged. This is
  the convention of llama.cpp and of most sampler configurations; it is not the
  mathematical reading of "keep zero tokens".

You may use your framework's sorting functions.

> [!WARNING]
> A built-in `topk` does not say which of several equal entries it returns, and the
> default sort of some frameworks is not stable. The ranking rule above is part of the
> specification.

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

Row 0 has no ties: the two largest logits are 3.0 (token 1) and 2.0 (token 3).

Row 1 has three logits equal to 2.0, at tokens 1, 3, and 4, and only two places. The
ranking is 1, 3, 4 (by index), then token 0, then the masked token 2. Tokens 1 and 3
are kept; token 4 is removed although its logit equals the smallest kept one. Keeping
"everything greater than or equal to the k-th largest value" would keep three tokens
here.

With `k = 4`, row 1 would keep tokens 1, 3, 4, and 0, and the masked token 2 would
stay `-inf`. With `k = 0`, `k = 5`, or `k = 100`, both rows come back unchanged.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$, $0 \le k \le 10^6$. Finite logits lie in
  $[-10^5, 10^5]$.
- The function only compares and copies values, so the comparison with the reference
  is effectively exact: kept entries must equal the input (relative tolerance `1e-4`,
  absolute `1e-6`), and removed entries must be exactly `-inf`.
- Ties occur in most tests, including ties at the boundary and rows whose logits are
  all equal.
- No Python loop over rows or tokens is needed.

## Why this matters

Top-k is the first filter in most sampler chains (llama.cpp applies it with `k = 40`
by default), and it is often there for speed as much as for quality: the samplers
after it then work on 40 candidates instead of a 150,000-token vocabulary. Its bugs
are about the boundary. A threshold comparison keeps every token tied with the k-th
one, which is harmless for typical float logits and wrong for quantized models,
masked rows, and padded vocabularies, where large groups of exactly equal logits are
normal; the later stages then receive more candidates than they were sized for. And
`k = 0` reaches this function from every client that leaves top-k unset, so an
implementation that reads it literally masks the whole vocabulary and the sampler
has nothing left to pick.
