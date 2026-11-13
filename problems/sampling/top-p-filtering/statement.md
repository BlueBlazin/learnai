Implement nucleus (top-p) filtering for a batch of logits: keep, in each row, the
smallest set of most probable tokens whose total probability reaches `p`, and return
the renormalized distribution over that set. The work is in the boundary (which token
is the last one kept) and in getting the result back from sorted order to token order,
for every row at once.

## Specification

```python
def top_p_filter(logits, p): ...
```

Each row of `logits` is treated independently. For one row, let
$\pi = \operatorname{softmax}(\text{logits})$ and rank the tokens by decreasing
probability. **Tokens with equal probability are ranked by token index, lowest first.**
Write $\sigma(1), \sigma(2), \ldots, \sigma(V)$ for the tokens in rank order and

$$
c_k = \sum_{j=1}^{k} \pi_{\sigma(j)}, \qquad c_0 = 0
$$

for the mass of the $k$ highest-ranked tokens. The nucleus is the shortest prefix of
the ranking whose mass reaches `p`:

$$
k^\ast = \min \{\, k : c_k \ge p \,\}, \qquad
\text{nucleus} = \{\sigma(1), \ldots, \sigma(k^\ast)\}.
$$

Equivalently, the token at rank $k$ is kept exactly when $c_{k-1} < p$: the mass of
the tokens ranked *before* it is still short of `p`. Two consequences:

- The token whose probability carries the cumulative mass to or past `p` **is kept**.
- The top-ranked token is always kept, because $c_0 = 0 < p$. At least one token
  survives however small `p` is.

The result holds the renormalized probabilities, in the original token order:

$$
\text{out}_i =
\begin{cases}
\pi_i / c_{k^\ast} & \text{if token } i \text{ is in the nucleus,} \\
0 & \text{otherwise.}
\end{cases}
$$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (a masked token,
  probability 0); every row has at least one finite entry.
- `p`: a Python float with $0 < p \le 1$, the same for every row.
- Returns a float32 tensor of shape `(B, V)`. Every row is non-negative and sums to 1.

You may use your framework's built-in `softmax`, sort, and cumulative sum. Do not loop
in Python over rows or tokens.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| result | `(B, V)` | float32 |

## Example

```python
# softmax(logits) = [[0.1, 0.4, 0.2, 0.3],
#                    [0.7, 0.1, 0.1, 0.1]]
logits = [[0.0, 1.3863, 0.6931, 1.0986],
          [1.9459, 0.0, 0.0, 0.0]]

top_p_filter(logits, 0.6)
# [[0.0, 0.5714, 0.0, 0.4286],
#  [1.0, 0.0,    0.0, 0.0   ]]
```

Row 0, ranked: token 1 (0.4), token 3 (0.3), token 2 (0.2), token 0 (0.1). The
cumulative masses are 0.4, 0.7, 0.9, 1.0. Token 1 alone has 0.4, which is short of
0.6; token 3 crosses it (0.7), so it is the last token kept. The nucleus
$\{1, 3\}$ has mass 0.7, and renormalizing gives $0.4 / 0.7 = 0.5714$ and
$0.3 / 0.7 = 0.4286$, written back to positions 1 and 3.

Row 1: token 0 alone has 0.7, which already reaches 0.6. It is the only survivor and
gets probability 1. The two rows keep different numbers of tokens.

A tie at the boundary: for probabilities `[0.25, 0.25, 0.25, 0.25]` and `p = 0.6`, the
ranking is 0, 1, 2, 3 (by index), the masses are 0.25, 0.5, 0.75, and the result is
`[0.3333, 0.3333, 0.3333, 0.0]`.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$, $0 < p \le 1$. Finite logits lie in
  $[-10^4, 10^4]$.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`.
- Whether a token is kept depends on comparing a partial sum $c_k$ with `p`, and
  float32 sums carry rounding error. The tests never make that comparison close: every
  $c_k$ with $k < V$ differs from `p` by at least `1e-4`. (For `p = 1.0` this means every
  token has probability of at least `1e-4`, and all of them are kept.)
- Ties in probability do occur in the tests, and the ranking rule above decides them.
  Equal logits always give exactly equal probabilities.

## Why this matters

Top-p is the default sampler of most LLM APIs, and this function runs once per
generated token for every sequence in the batch, between the model's forward pass and
the random draw. Its bugs are quiet. Dropping the token that crosses the threshold
keeps less mass than requested and, when the top token alone exceeds `p`, can leave
nothing to sample from. Writing probabilities back to the wrong positions still
produces a valid distribution, just over the wrong tokens, and the model's output
degrades without any error. A cutoff shared across the batch makes one request's
output depend on whatever else was batched with it.
