Implement min-p filtering for a batch of logits: in every row, remove the tokens whose
probability is less than a fraction `min_p` of the most likely token's probability.
Unlike top-k and top-p, the cutoff is relative to how confident the model is, so the
one thing to get right is what the fraction is a fraction *of*.

## Specification

```python
def min_p_filter(logits, min_p): ...
```

Each row of `logits` is treated independently. For one row, let
$\pi = \operatorname{softmax}(\text{logits})$ and $\pi_{\max} = \max_j \pi_j$. Token
$i$ is kept exactly when

$$
\pi_i \;\ge\; \text{min\_p} \cdot \pi_{\max},
$$

and the result is the logits with every other token masked:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if } \pi_v \ge \text{min\_p} \cdot \pi_{\max} \text{ in row } b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf` (a masked token,
  probability 0); every row has at least one finite entry. `+inf` and NaN never appear.
- `min_p`: a Python float with $0 \le \text{min\_p} \le 1$, the same for every row.
- Returns a float32 tensor of shape `(B, V)`: logits, not probabilities. A kept entry
  has exactly its input value; a removed entry is exactly `-inf`.

Consequences of the definition:

- The comparison is `>=`, so **the most likely token always survives**, together with
  every token tied with it. With `min_p = 1` those are the only survivors.
- With `min_p = 0` every token is kept and the logits are returned unchanged. A masked
  entry is `-inf` in the input and stays `-inf`.
- $\pi_{\max}$ is the largest probability of **that row**. The softmax is over all
  entries of the row, masked ones included (they contribute 0).
- The number of survivors differs from row to row, and nothing is renormalized: the
  kept logits are passed through untouched for the next sampler stage.

You may use your framework's built-in `softmax`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| result | `(B, V)` | float32 |

## Example

```python
# softmax(logits) = [[0.60, 0.30, 0.05, 0.03, 0.02, 0.00],
#                    [0.18, 0.18, 0.17, 0.16, 0.16, 0.15]]
logits = [
    [-0.5108, -1.2040, -2.9957, -3.5066, -3.9120, -inf],
    [-1.7148, -1.7148, -1.7720, -1.8326, -1.8326, -1.8971],
]

min_p_filter(logits, 0.2)
# [[-0.5108, -1.2040,    -inf,    -inf,    -inf,    -inf],
#  [-1.7148, -1.7148, -1.7720, -1.8326, -1.8326, -1.8971]]
```

Row 0 is confident: its top probability is 0.60, so the cutoff is
$0.2 \times 0.60 = 0.12$. Tokens 0 and 1 pass; the three tokens at 0.05 and below are
removed, and the masked token stays masked.

Row 1 is flat: its top probability is 0.18, so the cutoff is $0.2 \times 0.18 = 0.036$
and all six tokens pass. Note that every probability in this row is below 0.2: using
`min_p` as an absolute cutoff would remove the whole row.

The same `min_p` kept two tokens in one row and six in the other. Sampling from the
result means taking its softmax: row 0 becomes `[0.6667, 0.3333, 0, 0, 0, 0]`.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$. `min_p` is either exactly `0` or lies in
  $[10^{-4}, 1]$. Finite logits lie in $[-1000, 1000]$.
- Kept entries are compared with the reference to a relative tolerance of `1e-4` and
  an absolute tolerance of `1e-6` (they are copies of the input); removed entries must
  be exactly `-inf`.
- The test $\pi_i \ge \text{min\_p} \cdot \pi_{\max}$ is equivalent to
  $x_i \ge x_{\max} + \ln(\text{min\_p})$ on the logits $x$, and you may implement
  either. Float32 rounds the two differently, so the tests never make the comparison
  close: every finite logit differs from $x_{\max} + \ln(\text{min\_p})$ by at least
  `1e-3`. The one exception is exact in both forms: with `min_p = 1`, logits equal to
  the row maximum are kept.
- No Python loop over rows or tokens is needed.

## Why this matters

Min-p (Nguyen et al., 2024) is the truncation sampler that local inference engines
reach for first: llama.cpp enables it by default with `min_p = 0.05`, and mlx-lm,
Hugging Face `transformers`, and vLLM all ship it. Its appeal is that one setting
behaves sensibly in both regimes. When the model is sure, a fixed top-k or top-p still
lets a long tail of unlikely tokens through, which is where incoherent output at high
temperature comes from; when the model is unsure, they cut off reasonable candidates.
A cutoff proportional to the top probability does neither. That property is the entire
method, and it is lost by the two most common ways of getting it wrong: an absolute
cutoff (which empties flat rows, so the sampler has nothing to choose from) and a top
probability shared across the batch (which makes one request's output depend on how
confident the model is about a neighbor's).
