## The idea

Lay a row's probabilities $p_0, \ldots, p_{V-1}$ end to end on the interval
$[0, C)$, where $C = c_{V-1}$ is the row's total. With $c_{-1} = 0$, token $k$ owns
the sub-interval $[c_{k-1}, c_k)$ of length $p_k$. A point $t$ drawn uniformly from
$[0, C)$ lands in token $k$'s interval with probability $p_k / C$. The interval
containing $t$ is the first one whose right endpoint strictly exceeds $t$:

$$
\text{token} = \min \{\, k : c_k > t \,\}, \qquad t = u \cdot C .
$$

This evaluates the inverse of the cumulative distribution function at $u$. Both edge
rules follow directly from the geometry rather than special cases:

- A token with $p_k = 0$ owns an empty interval $[c_{k-1}, c_{k-1})$, so no point
  can land in it. The strict inequality enforces this because $c_k = c_{k-1}$, so
  whenever $c_k > t$ holds, $c_{k-1} > t$ already holds and $k$ is not the minimum.
- Because $u < 1$, we have $t = u \cdot C < C = c_{V-1}$, so at least one index
  satisfies $c_k > t$. Scaling by $C$ ensures the intervals cover $[0, C)$ rather
  than $[0, 1)$. If $C = 0.999$ and you compare against $u$ directly, one draw in a
  thousand falls past every interval.

## Reference solution (PyTorch)

```python
def sample(probs, u):
    cdf = torch.cumsum(probs, dim=-1)
    threshold = u[:, None] * cdf[:, -1:]
    return (cdf <= threshold).sum(dim=-1)
```

- Using `cdf[:, -1:]` as the row total guarantees that the threshold is scaled by
  the exact float that appears in the last column of `cdf`. A separate
  `probs.sum(-1)` sums the entries in a different order and can differ in the last
  bit, which can break `threshold < cdf[:, -1]`.
- For every float32 value `u < 1`, the product `u * C` rounds to a float32 strictly
  below `C`. The largest float32 below 1 is $1 - 2^{-24}$, and $C \cdot 2^{-24}$ is
  at least half the spacing of floats just below `C`.
- Because `cdf` is non-decreasing, counting how many entries satisfy
  `cdf <= threshold` gives the exact index of the first entry that exceeds
  `threshold`. This avoids `argmax`, `searchsorted`, and any fallback logic.
- Indexing `u[:, None]` gives `u` shape `(B, 1)` so each row gets its own threshold.

Using `searchsorted(cdf, threshold, right=True)` performs the same lookup via binary
search in $O(\log V)$ time per row after the cumulative sum, where `right=True`
enforces the strict inequality.

## Common mistakes

- **Using `>=` instead of `>`** (or counting `cdf < threshold`, or calling
  `searchsorted` with its default `side="left"`). When `u = 0`, the threshold is 0
  and $c_0 \ge 0$ always holds, so token 0 is returned even when its probability is
  0. A 24-bit uniform generator produces `0.0` about once in 16.7 million draws.
- **Comparing against raw `u`.** When a row's total is below `u`, no cumulative sum
  exceeds `u` and the count is `V`. That out-of-range token ID fails one step later
  during the embedding lookup of the next forward pass.
- **Clamping the index to `V - 1`.** Clamping hides the out-of-range crash by
  returning the last token of the vocabulary regardless of its probability, which is
  0 after top-k or top-p filtering.
- **Taking `argmax` of the mask `cdf > u`.** When every entry is `False`, `argmax`
  returns 0, creating the same silent bug on token 0 instead of token `V - 1`.
- **Sharing one uniform across the batch** (such as `u[0]` or a single scalar). All
  rows in the batch then share the same draw, so identical prompts produce identical
  continuations and batch samples are no longer independent.
- **Reparameterizing with `1 - u`.** The marginal distribution stays correct, so a
  statistical test alone would pass, but the function violates the specification for
  a given `u` and breaks reproducibility across implementations.
- **Drawing randomness inside the function** (such as calling `torch.multinomial`).
  This breaks determinism, cross-framework reproducibility, and replay from logged
  seeds.

## Complexity

$O(B \cdot V)$ time and memory for one cumulative sum and one elementwise
comparison. Using binary search does not change the overall complexity because the
cumulative sum is already linear in $V$.

## In the wild

`llama.cpp` uses this algorithm in its final sampler (`llama_sampler_dist_apply`).
It draws a uniform number, multiplies it by the running total of candidate
probabilities (`sum_tgt = sum_cum * rnd`), walks the candidates until the running
sum reaches the target, and falls back to the last candidate if the loop ends
without a hit. Because its candidates are already filtered, zero-probability entries
have already been removed.

`jax.random.categorical` and `mx.random.categorical` take logits and use the
Gumbel-max trick instead (add independent Gumbel noise and take the argmax). That
approach avoids both the cumulative sum and normalization, at the cost of $V$ random
numbers per sample (covered in `gumbel-max-sample`). Inference engines that must
reproduce a token stream exactly, such as for speculative decoding or deterministic
testing, prefer inverse-CDF sampling because it consumes exactly one uniform number
per token.
