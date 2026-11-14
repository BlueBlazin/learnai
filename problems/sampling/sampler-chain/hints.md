Every filter decides from the softmax of the logits it is handed. Take a row and a
top-p of 0.9. Is the set of tokens holding 90% of the mass the same before and after
you divide the logits by 2? What about before and after you subtract a penalty from
the most likely token? Each stage's answer depends on what ran before it, so write
the stages as a pipeline in which each one consumes the previous one's output, in
the order the statement gives.

---

Because the kit's filters return logits with `-inf` for removed tokens, they chain
directly, and "the distribution renormalized over the survivors" needs no code:
`softmax` of logits with `-inf` entries already is that distribution. So stages 2 to
4 are three calls, each on the result of the one before. Two details: the penalties
need a per-row count of every vocabulary entry in `history` (compare
`history[:, :, None]` with `arange(V)` and sum over the time axis), and a `top_p` of
exactly `1.0` must skip the filter instead of calling it.

---

```
counts = (history[:, :, None] == arange(V)).sum(axis=1)          # (B, V)
seen   = counts > 0
x = where(seen, where(logits > 0, logits / rep, logits * rep), logits)
x = x - counts * freq - seen * pres
x = ops.top_k_filter(x, top_k)
if top_p < 1: x = ops.top_p_filter(x, top_p)
x = ops.min_p_filter(x, min_p)
x = x / temperature                                               # chain_logits
```

For the draw, `probs = ops.softmax(x)` and `cdf = cumsum(probs)`; the token is the
number of entries with `cdf <= u[:, None] * cdf[:, -1:]`.
