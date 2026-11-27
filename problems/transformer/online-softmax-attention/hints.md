Suppose you maintain two running sums per query over the keys seen so far: $\sum_j e^{s_j}$ and $\sum_j e^{s_j} v_j$. Their ratio gives exact attention over those keys, and each new block adds terms to both sums. That is already a valid single-pass algorithm mathematically. What breaks in `float32`, and how did `stable-softmax` fix the same issue when all scores were available at once?

---

Store both sums shifted by $m$, the largest score the query has seen so far: `norm` $= \sum_j e^{s_j - m}$ and `acc` $= \sum_j e^{s_j - m} v_j$. Every exponent is at most 0, so nothing overflows, and the maximum term equals 1, so `norm` is at least 1. When a new block raises the maximum from $m$ to $m'$, multiply both existing sums by $e^{m - m'}$ before adding the new block's terms shifted by $m'$. The ratio `acc / norm` cancels the shift, so you never need $e^m$ itself.

---

Initialize `m = -inf`, `norm = 0`, and `acc = 0` with one row per query. For each block:

```
s = where(mask, q @ k.T / sqrt(D), -inf)
m_new = maximum(m, max(s, axis=-1, keepdims=True))
shift = where(isinf(m_new), 0, m_new)        # guard for all-masked prefix
rescale = exp(m - shift)
w = exp(s - shift)
norm = norm * rescale + sum(w, axis=-1, keepdims=True)
acc = acc * rescale + w @ v
m = m_new
```

The `shift` guard protects queries that have not yet seen a visible key. For those queries, `m_new` is `-inf`, so `s - m_new` would evaluate to `-inf - (-inf)` (`NaN`). Setting `shift` to `0` makes every masked term `exp(-inf) = 0`. On the query's first visible block, `rescale` is `exp(-inf - shift) = 0`, which safely multiplies the zero sums. Iterate with a plain `for k, v, mask in kv_blocks:` loop and do not store blocks in a list, so each iteration releases the previous block.
