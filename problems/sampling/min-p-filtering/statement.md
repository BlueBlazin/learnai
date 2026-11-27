Implement min-p filtering for a batch of logits. For each row, mask out every token
whose probability is less than `min_p` times the probability of that row's most
likely token.

![This problem is Min-p, which removes tokens far less probable than the top one.](assets/map.svg)

## Specification

```python
def min_p_filter(logits, min_p): ...
```

The input `logits` contains `B` rows over a vocabulary of `V` tokens. Entries may be
`-inf` to represent already-masked tokens with probability 0, and every row has at
least one finite entry. `+inf` and NaN never appear. The threshold factor `min_p` is
a Python float with $0 \le \text{min\_p} \le 1$, shared across all rows.

Each row $b$ is filtered independently. Let $\pi = \operatorname{softmax}(\text{logits}[b])$
be the row's probabilities (masked `-inf` entries contribute probability 0), and let
$\pi_{\max} = \max_j \pi_j$ be the largest probability in that row. Token $v$ is
kept when

$$
\pi_v \;\ge\; \text{min\_p} \cdot \pi_{\max}.
$$

The output has the same shape as `logits`. Kept tokens retain their exact input
logit values, and removed tokens are replaced with `-inf`:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if } \pi_v \ge \text{min\_p} \cdot \pi_{\max} \text{ in row } b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

The following properties follow from this definition:

- Because the comparison is `>=`, **the most likely token always survives**, along
  with every token tied with it. When `min_p = 1`, those are the only survivors.
- When `min_p = 0`, every token is kept and the input logits are returned unchanged.
  Existing `-inf` entries remain `-inf`.
- $\pi_{\max}$ is computed per row over all `V` entries of that row.
- The number of surviving tokens can vary from row to row. Return unnormalized
  logits (not probabilities) so kept values pass untouched to the next sampler
  stage.

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

Row 0 is confident, with a top probability of 0.60, so its cutoff is
$0.2 \times 0.60 = 0.12$. Tokens 0 and 1 pass, the three tokens at 0.05 and below
are masked to `-inf`, and the already-masked token stays `-inf`.

Row 1 is flat, with a top probability of 0.18, so its cutoff is
$0.2 \times 0.18 = 0.036$ and all six tokens pass. Every probability in this row is
below 0.2, so an absolute cutoff of 0.2 would have removed the entire row.

The same `min_p` keeps two tokens in row 0 and six in row 1. Taking the softmax of
the filtered output renormalizes row 0 to `[0.6667, 0.3333, 0, 0, 0, 0]`.

## Constraints

- $1 \le B \le 64$ and $1 \le V \le 2000$. `min_p` is either `0` or lies in
  $[10^{-4}, 1]$. Finite logits lie in $[-1000, 1000]$.
- Kept entries are compared with the reference using a relative tolerance of `1e-4`
  and an absolute tolerance of `1e-6`. Removed entries must be exactly `-inf`.
- The probability test $\pi_v \ge \text{min\_p} \cdot \pi_{\max}$ is equivalent to
  $x_v \ge x_{\max} + \ln(\text{min\_p})$ on a row's logits $x$ with row maximum
  $x_{\max}$, and you may implement either form. Because float32 rounds the two
  differently, every finite test logit differs from $x_{\max} + \ln(\text{min\_p})$
  by at least `1e-3`. The only exception is `min_p = 1`, where logits equal to the
  row maximum are kept in both forms.
- No Python loop over rows or tokens is needed.

## Why this matters

Min-p (Nguyen et al., 2024) is a standard truncation sampler in modern inference
engines. `llama.cpp` enables it by default with `min_p = 0.05`, and `mlx-lm`,
Hugging Face `transformers`, and `vLLM` all ship it.

A single `min_p` setting adapts to how confident the model is. When the model is sure,
fixed top-k or top-p can still admit a long tail of unlikely tokens that cause
incoherent output at high temperature. When the model is unsure, they can cut off
reasonable candidates. Scaling the cutoff by the top probability avoids both
problems.

Two common bugs destroy this property. Using an absolute probability cutoff empties
flat rows so the sampler has nothing to choose from. Taking the maximum probability
across the entire batch couples unrelated requests, making one prompt's output
depend on how confident the model is on another prompt in the same batch.
