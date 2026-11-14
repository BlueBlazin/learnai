## The idea

Lay the probabilities end to end on the interval $[0, C)$, where $C$ is the row's
total. Token $k$ owns the piece $[c_{k-1}, c_k)$, whose length is $p_k$. A point $t$
uniform on $[0, C)$ lands in piece $k$ with probability $p_k / C$, and the piece it
lands in is the first one whose right end is beyond it:

$$
\text{token} = \min \{\, k : c_k > t \,\}, \qquad t = u \cdot C .
$$

This is the inverse of the cumulative distribution function, evaluated at $u$. Both
edge rules of the problem are properties of this picture, not special cases:

- A token with $p_k = 0$ owns an empty piece $[c_{k-1}, c_{k-1})$. No point lands in
  it. With a strict inequality the formula agrees: $c_k = c_{k-1}$, so if $c_k > t$
  then already $c_{k-1} > t$ and $k$ is not the minimum.
- $t = u \cdot C < C = c_{V-1}$ for every $u < 1$, so the set is never empty. The
  pieces must cover $[0, C)$, not $[0, 1)$: if $C = 0.999$ and the point is drawn on
  $[0, 1)$, one draw in a thousand lands beyond every piece.

## Reference solution (PyTorch)

```python
def sample(probs, u):
    cdf = torch.cumsum(probs, dim=-1)
    threshold = u[:, None] * cdf[:, -1:]
    return (cdf <= threshold).sum(dim=-1)
```

- The total is `cdf[:, -1:]`, the same float that the comparison uses. A separately
  computed `probs.sum(-1)` adds the same numbers in a different order and may differ
  from the last cumulative sum in the last bit, which is enough to break
  `threshold < cdf[:, -1]`.
- `u * C` rounds to a float32 strictly below `C` for every float32 `u < 1`: the
  largest such `u` is $1 - 2^{-24}$, and $C \cdot 2^{-24}$ is at least half the
  spacing of floats just below `C`.
- Counting `cdf <= threshold` gives the first index that exceeds the threshold,
  because `cdf` is non-decreasing. It needs no `argmax`, no `searchsorted`, and has
  no "nothing found" case.
- `u[:, None]` gives each row its own threshold.

`searchsorted(cdf, threshold, right=True)` is the same computation with a binary
search, $O(\log V)$ per row after the cumulative sum. `right=True` is the strict
inequality.

## Common mistakes

- **`>=` instead of `>`** (or counting `cdf < threshold`, or `searchsorted` with its
  default `side="left"`). At `u = 0` the threshold is 0 and $c_0 \ge 0$ always holds,
  so token 0 is returned even when its probability is 0. `u = 0` is not exotic: a
  generator of 24-bit uniforms returns it once in 16.7 million draws.
- **Comparing with the raw `u`.** If the row's total is below `u`, nothing exceeds it
  and the count is `V`. The token id is out of range, and the failure appears one
  step later, in the embedding lookup of the next forward pass.
- **Clamping the index to `V - 1`.** It removes the crash and returns the last token
  of the vocabulary, whatever its probability. After top-k or top-p that probability
  is 0.
- **`argmax` of the mask `cdf > u`.** When no entry is `True`, `argmax` returns 0:
  the same bug, now on the first token.
- **One uniform for the whole batch** (`u[0]`, or a scalar drawn once). Every row of
  the batch is then driven by the same number: identical prompts give identical
  continuations, and the samples are not independent.
- **Using `1 - u`**, or any other reparameterization. The distribution of the output
  is still right, which is why a statistical test alone would accept it, but the
  function no longer computes what the specification says for a given `u`, and
  results stop being reproducible across implementations.
- **Drawing the randomness inside** (`torch.multinomial`). The function is then not
  testable, not reproducible across frameworks, and cannot be replayed from a logged
  seed.

## Complexity

$O(B \cdot V)$ time and memory: one cumulative sum and one comparison. The binary
search variant does not change the total, since the cumulative sum is already linear.

## In the wild

llama.cpp's final sampler (`llama_sampler_dist_apply`) is this algorithm: it draws a
uniform number, multiplies it by the running total of the candidate probabilities
(`sum_tgt = sum_cum * rnd`), walks the candidates accumulating probability until the
running sum reaches the target, and falls back to the last candidate if the loop ends
without a hit. Its candidates have already been filtered, so zero-probability entries
are gone before this point. `jax.random.categorical` and `mx.random.categorical` take
logits and use the Gumbel-max trick instead (add independent Gumbel noise, take the
argmax), which needs $V$ random numbers per sample but no cumulative sum and no
normalization; `gumbel-max-sample` covers it. Engines that must reproduce a stream of
tokens exactly, for speculative decoding or for tests like these, prefer inverse-CDF
sampling because it consumes exactly one uniform number per token.
