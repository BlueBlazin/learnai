Implement top-n-sigma filtering for a batch of logits: in every row, keep the tokens
whose logit lies within `n` standard deviations of the row's largest logit, and mask
the rest. The filter works on raw logits, with no softmax, so the whole exercise is
the statistic: which entries the standard deviation is taken over, and with which
definition.

## Specification

```python
def top_n_sigma_filter(logits, n): ...
```

Each row of `logits` is treated independently. For one row $x$, let $F$ be the set of
its **finite** entries (a masked token is `-inf` and is not in $F$), and

$$
M = \max_{i \in F} x_i, \qquad
\mu = \frac{1}{|F|} \sum_{i \in F} x_i, \qquad
\sigma = \sqrt{\frac{1}{|F|} \sum_{i \in F} (x_i - \mu)^2}\,.
$$

$\sigma$ is the **population** standard deviation (the divisor is $|F|$, not
$|F| - 1$) of the finite logits. Token $i$ is kept exactly when

$$
x_i \;\ge\; M - n\,\sigma ,
$$

and the result is the logits with every other token masked:

$$
\text{out}[b, v] =
\begin{cases}
\text{logits}[b, v] & \text{if } \text{logits}[b, v] \ge M_b - n\,\sigma_b, \\
-\infty & \text{otherwise.}
\end{cases}
$$

- `logits`: a float32 tensor of shape `(B, V)`. Entries may be `-inf`; every row has at
  least one finite entry. `+inf` and NaN never appear.
- `n`: a Python float, the same for every row. **If `n <= 0` the filter is disabled**
  and the logits are returned unchanged (the convention of llama.cpp, whose default is
  `-1`).
- Returns a float32 tensor of shape `(B, V)`: logits, not probabilities. A kept entry
  has exactly its input value; a removed entry is exactly `-inf`.

Consequences of the definition:

- The comparison is `>=`, so the largest logit of a row always survives, together
  with every entry tied with it. When $\sigma = 0$ (one finite entry, or all finite
  entries equal) those are the only candidates, and all of them are kept.
- A masked entry stays `-inf`, and it changes neither $\mu$ nor $\sigma$: a row gives
  the same result for its finite entries whether or not masked entries are present.
- $M$, $\mu$, and $\sigma$ belong to one row. Rows of very different spread sit side
  by side in one batch.
- Nothing is renormalized.

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

Row 0 has five finite entries with mean 0 and
$\sigma = \sqrt{(4 + 1 + 0 + 1 + 4) / 5} = 1.4142$. The cutoff is
$2 - 1.4142 = 0.5858$: the logits 2 and 1 pass. The masked entry is not one of the
five; counting it would make the mean `-inf` and the standard deviation NaN.

Row 1 has mean $10 / 6 = 1.6667$ and
$\sigma = \sqrt{(8.3333^2 + 5 \cdot 1.6667^2) / 6} = 3.7268$. The cutoff is
$10 - 3.7268 = 6.2732$, and only the outlier passes: five tokens at the same level are
"noise", one token far above them is the signal.

## Constraints

- $1 \le B \le 64$, $1 \le V \le 2000$. Finite logits lie in $[-100, 100]$, and
  $n \le 10$.
- Kept entries are compared with the reference to a relative tolerance of `1e-4` and
  an absolute tolerance of `1e-6` (they are copies of the input); removed entries must
  be exactly `-inf`.
- Float32 rounding cannot decide a test: every finite logit differs from the cutoff
  $M - n\sigma$ by at least `1e-3`. The one exception is exact: when $\sigma = 0$, the
  entries equal to $M$ are at the cutoff and are kept.
- No Python loop over rows or tokens is needed.

## Why this matters

Top-n-sigma (Tang et al., 2024) is the newest of the truncation samplers in llama.cpp
(`--top-nsigma`, ahead of top-k in its default sampler order, off until you set it). Its selling
point is that it cuts in logit space: the kept set is the same at every temperature,
where top-p and min-p let more of the tail through as the temperature rises. That
property holds only if the statistic is computed as specified. By the time this
sampler runs, a grammar, a logit bias, or an earlier filter has usually set some logits
to `-inf`; an implementation that feeds those to a library `std` gets NaN, every
comparison with NaN is false, and the sampler silently removes the entire vocabulary.
