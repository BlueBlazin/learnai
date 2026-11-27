Every filter decides from the softmax of the logits it receives. Take a row of logits
and a `top_p` of `0.9`. Does the set of tokens holding 90% of the mass stay the same
after you divide the logits by `2`? Does it stay the same after you penalize the top
token? Because each stage depends on what ran before it, write the stages as a pipeline
where each one consumes the previous stage's output in the exact order given in the
specification.

---

The kit's filters return logits with `-inf` for removed tokens, so they chain directly.
You do not need to write separate renormalization code because calling `softmax` on
logits with `-inf` entries already produces the distribution renormalized over the
survivors. That makes stages 2 to 4 three sequential calls. Keep two details in mind.
First, compute per-row token counts from `history` by comparing `history[:, :, None]`
with `arange(V)` and summing over the time axis. Second, skip `top_p_filter` when
`top_p == 1.0` instead of calling it.

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

For `sample_chain`, compute `probs = ops.softmax(x)` and `cdf = cumsum(probs)`. The
sampled token index is the number of entries satisfying `cdf <= u[:, None] * cdf[:, -1:]`.
