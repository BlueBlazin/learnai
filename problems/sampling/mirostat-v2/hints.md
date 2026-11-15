Every quantity in the step is a function of one tensor, the log-probabilities of the
row. Which three places use them: the truncation, the sampling, and the update? And
which of the three must not see the effect of the truncation?

---

Compute `log_p = logits - logsumexp(logits)` once, and `surprise = -log_p / ln 2`.
The kept set is `surprise <= mu[:, None]`, together with the argmax of the row, which
covers the case where nothing passes (when something passes, the argmax passes too,
because it is the least surprising token). Sampling is `categorical-sample` on the
probabilities with the removed entries set to zero: a cumulative sum, a threshold of
`u` times the last cumulative sum, and a count.

---

```
log_p    = logits - logsumexp(logits, axis=-1, keepdims=True)
surprise = -log_p / ln(2)                                   # bits; +inf where masked
keep     = (surprise <= mu[:, None]) | (arange(V) == argmax(logits, keepdims=True))
cdf      = cumsum(where(keep, exp(log_p), 0), axis=-1)
token    = sum(cdf <= u[:, None] * cdf[:, -1:], axis=-1)
observed = take_along_axis(surprise, token[:, None], axis=-1)[:, 0]
new_mu   = mu - eta * (observed - tau)
```

`observed` is read from `surprise`, the tensor computed before the truncation. If
you renormalize the kept probabilities and take the logarithm of those, the update
sees a smaller number than the statement asks for.
