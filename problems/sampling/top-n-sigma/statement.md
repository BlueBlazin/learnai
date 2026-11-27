Implement top-n-sigma filtering for a batch of raw token logits. In each row, keep every token whose logit is within `n` standard deviations of the row maximum, and mask all other tokens to `-inf`.

## Specification

```python
def top_n_sigma_filter(logits, n): ...
```

The input `logits` is a 2D tensor with one row per sequence in a batch of size `B` and one column per token in a vocabulary of size `V`. Entries may be `-inf` to indicate tokens already masked by an earlier stage, and every row has at least one finite entry. Neither `+inf` nor NaN appears in the input.

The parameter `n` is a Python float shared across all rows. If `n <= 0`, the filter is disabled and returns `logits` unchanged. This matches the convention in `llama.cpp`, where the default is `-1`.

When `n > 0`, process each row `b` independently. Let $F_b$ be the set of token indices $v$ where $\text{logits}[b, v]$ is finite. Compute the maximum $M_b$, the mean $\mu_b$, and the population standard deviation $\sigma_b$ over the finite entries in row `b`:

$$
M_b = \max_{v \in F_b} \text{logits}[b, v], \qquad
\mu_b = \frac{1}{|F_b|} \sum_{v \in F_b} \text{logits}[b, v], \qquad
\sigma_b = \sqrt{\frac{1}{|F_b|} \sum_{v \in F_b} \big(\text{logits}[b, v] - \mu_b\big)^2}\,.
$$

Note that $\sigma_b$ uses the population divisor $|F_b|$, not $|F_b| - 1$. Keep token $v$ if its logit meets or exceeds the cutoff $M_b - n\,\sigma_b$, and mask it to `-inf` otherwise:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if } \text{logits}[b, v] \ge M_b - n\,\sigma_b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

The function returns raw logits rather than probabilities, with no renormalization. Kept entries preserve their exact input values, and removed entries become `-inf`.

A few edge cases follow directly from this definition:

- Because the comparison is `>=`, the largest logit in each row always survives along with any tokens tied with it. When $\sigma_b = 0$ (one finite entry, or all finite entries equal), every finite entry equals the maximum and is kept.
- Masked `-inf` entries remain `-inf` and are excluded from $\mu_b$ and $\sigma_b$. Adding or removing `-inf` entries does not change the result for the finite entries.
- Statistics $M_b$, $\mu_b$, and $\sigma_b$ are computed per row, so rows with very different spreads can appear in the same batch.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, V)` | float32 |
| result | `(B, V)` | float32 |

## Example

```python
logits = [
    [2.0, 1.0, 0.0, -1.0, -2.0, -inf],
    [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

top_n_sigma_filter(logits, 1.0)
# [[ 2.0,  1.0, -inf, -inf, -inf, -inf],
#  [10.0, -inf, -inf, -inf, -inf, -inf]]
```

Row 0 has five finite entries with mean `0.0` and $\sigma = \sqrt{(4 + 1 + 0 + 1 + 4) / 5} = 1.4142$. The cutoff is $2.0 - 1.4142 = 0.5858$, so the logits `2.0` and `1.0` pass. The `-inf` entry is excluded from the statistics. Including it would make the mean `-inf` and the standard deviation NaN.

Row 1 has six finite entries with mean $10 / 6 = 1.6667$ and $\sigma = \sqrt{(8.3333^2 + 5 \cdot 1.6667^2) / 6} = 3.7268$. The cutoff is $10.0 - 3.7268 = 6.2732$, so only the `10.0` outlier passes while the five flat tokens are masked out.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$. Finite logits lie in $[-100, 100]$, and $n \le 10$.
- Kept entries are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. Removed entries must be exactly `-inf`.
- Every finite logit differs from the cutoff $M_b - n\,\sigma_b$ by at least `1e-3`, so float32 rounding cannot flip a boundary decision. The only exception is when $\sigma_b = 0$, where entries equal to $M_b$ sit exactly at the cutoff and are kept.
- No Python loop over rows or tokens is needed.

## Why this matters

Top-n-sigma (Tang et al., 2024) is a truncation sampler used in `llama.cpp` (`--top-nsigma`, placed before top-k in its default sampler order and disabled by default). Because it truncates in logit space rather than probability space, the kept set stays identical at every temperature. By contrast, top-p and min-p admit more of the tail as temperature rises.

In real inference pipelines, a grammar constraint, logit bias, or earlier filter often sets some logits to `-inf` before top-n-sigma runs. Passing those `-inf` values into a library `std` produces NaN. Every comparison against NaN evaluates to false, which silently masks out the entire vocabulary.
