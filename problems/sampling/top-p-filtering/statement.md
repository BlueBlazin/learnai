Implement nucleus (top-p) filtering for a batch of logits. For each row, convert the
logits to probabilities, keep the smallest set of most probable tokens whose total
probability is at least `p`, and return the renormalized probability distribution over
those tokens in the original token order.

![This problem is Top-p, which keeps the most probable tokens until their mass reaches $p$.](assets/map.svg)

## Specification

```python
def top_p_filter(logits, p): ...
```

`logits` contains unnormalized log-probabilities for a batch of `B` rows over a
vocabulary of `V` tokens, and `p` is a Python float with $0 < p \le 1$ shared across
every row. Entries of `logits` may be `-inf` (masked tokens with probability 0), and
every row has at least one finite entry.

Each row of `logits` is filtered independently. For one row, compute
$\pi = \operatorname{softmax}(\text{logits})$ and rank the `V` tokens by decreasing
probability. **Tokens with equal probability are ranked by token index, lowest first.**
Write $\sigma(1), \sigma(2), \ldots, \sigma(V)$ for the token indices in rank order, and
define the cumulative probability of the top $k$ tokens as

$$
c_k = \sum_{j=1}^{k} \pi_{\sigma(j)}, \qquad c_0 = 0.
$$

The nucleus is the shortest prefix of the ranking whose cumulative probability reaches
`p`:

$$
k^\ast = \min \{\, k : c_k \ge p \,\}, \qquad
\text{nucleus} = \{\sigma(1), \ldots, \sigma(k^\ast)\}.
$$

Equivalently, the token at rank $k$ is kept if and only if $c_{k-1} < p$, meaning the
tokens ranked *before* it have not yet reached mass `p`. This rule has two consequences:

- The token whose probability pushes the cumulative mass to or past `p` **is kept**.
- The top-ranked token is always kept because $c_0 = 0 < p$. At least one token survives
  regardless of how small `p` is.

Return a tensor of the same shape and dtype as `logits` containing the renormalized
probabilities in the original token order, so every row is non-negative and sums to 1:

$$
\text{out}_i =
\begin{cases}
\pi_i / c_{k^\ast} & \text{if token } i \text{ is in the nucleus,} \\
0 & \text{otherwise.}
\end{cases}
$$

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

In row 0, ranking tokens by decreasing probability gives token 1 (0.4), token 3 (0.3),
token 2 (0.2), and token 0 (0.1). Their cumulative probabilities are 0.4, 0.7, 0.9, and
1.0. Token 1 alone has mass 0.4, which is below 0.6. Adding token 3 brings the cumulative
mass to 0.7, crossing 0.6, so token 3 is the last token kept. Dividing by the nucleus
mass 0.7 gives $0.4 / 0.7 = 0.5714$ for token 1 and $0.3 / 0.7 = 0.4286$ for token 3,
written back to positions 1 and 3.

![Row 0 in rank order with $p = 0.6$: token 3 carries the cumulative mass past $p$ and is the last one kept.](assets/nucleus.svg)

In row 1, token 0 alone has probability 0.7, which already reaches 0.6. It is the only
surviving token and gets probability 1.0. Notice that different rows can keep different
numbers of tokens.

For a tie at the boundary, consider probabilities `[0.25, 0.25, 0.25, 0.25]` with
`p = 0.6`. Tied tokens are ordered by index (0, 1, 2, 3), giving cumulative masses 0.25,
0.5, 0.75, and 1.0. Tokens 0, 1, and 2 are kept, and renormalizing produces
`[0.3333, 0.3333, 0.3333, 0.0]`.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$, and $0 < p \le 1$. Finite logits lie in
  $[-10^4, 10^4]$.
- Results are compared against the reference with a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`.
- Float32 cumulative sums carry rounding error, so the tests keep threshold comparisons
  well-separated. Every partial sum $c_k$ with $k < V$ differs from `p` by at least
  `1e-4`. For `p = 1.0`, every token has probability at least `1e-4`, and all tokens are
  kept.
- Ties in probability occur in the tests and are resolved by the index-order rule above.
  Equal logits always give equal probabilities.

## Why this matters

Top-p filtering is the default sampling method in most LLM APIs. It runs once per
generated token for every sequence in the batch, between the model's forward pass and
the random draw.

Bugs in this step rarely raise errors. Dropping the token that crosses `p` leaves less
mass than requested, and it can zero out every token when the top token alone exceeds
`p`. Writing probabilities back to the wrong indices still yields a valid distribution,
just over the wrong tokens, silently degrading model outputs. Computing a single cutoff
shared across the batch makes one request's output depend on whatever else was batched
alongside it.
