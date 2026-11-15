Suppose you had processed some blocks and kept, for each query, two sums over the keys
seen so far: $\sum_j e^{s_j}$ and $\sum_j e^{s_j} v_j$. Their ratio is the answer for
those keys, and a new block just adds terms to both. That is already a correct
one-pass algorithm on paper. What goes wrong in float32, and what did `stable-softmax`
do about the same problem when all scores were available at once?

---

Keep the sums in units of $e^{m}$, where $m$ is the largest score the query has seen:
`norm` $= \sum_j e^{s_j - m}$ and `acc` $= \sum_j e^{s_j - m} v_j$. Every exponent is
at most 0, so nothing overflows, and the term of the maximum is 1, so `norm` is at
least 1. When a block raises the maximum from $m$ to $m'$, the old sums are in the
wrong unit: multiply both by $e^{m - m'}$ before adding the block's terms, which are
computed with $m'$. The ratio `acc / norm` is unaffected by the unit, so you never
need $e^{m}$ itself.

---

Start with `m = -inf`, `norm = 0`, `acc = 0`, one row per query. For each block:

```
s = where(mask, q @ k.T / sqrt(D), -inf)
m_new = maximum(m, max(s, axis=-1, keepdims=True))
shift = where(isinf(m_new), 0, m_new)        # the guard
rescale = exp(m - shift)
w = exp(s - shift)
norm = norm * rescale + sum(w, axis=-1, keepdims=True)
acc = acc * rescale + w @ v
m = m_new
```

The guard handles a query that has seen no visible key yet: its `m_new` is `-inf`,
and `s - m_new` would be `-inf - (-inf)`, NaN. With a shift of 0 its terms are
`exp(-inf) = 0`, as they should be, and at its first visible block `rescale` is
`exp(-inf - shift) = 0`, which discards nothing because the sums were 0. Iterate with
a plain `for k, v, mask in kv_blocks:` and keep no list: the loop variables are
rebound on each iteration, which is what releases the previous block.
