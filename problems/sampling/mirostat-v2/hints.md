Every quantity in the step is a function of one tensor, the log-probabilities of each
row. Those log-probabilities are used in three places: truncation, sampling, and the
$\mu$ update. Which of those three must not see the effect of truncation?

---

Compute `log_p = logits - logsumexp(logits)` once, and convert it to bits with
`surprise = -log_p / ln 2`. The kept mask is `surprise <= mu[:, None]` combined with
the argmax of each row. Including the argmax handles the empty-set fallback without a
branch, because whenever any token passes the threshold, the least surprising token
passes too. Sampling is `categorical-sample` on the probabilities with dropped
entries zeroed out. Compute the cumulative sum, scale `u` by the final cumulative sum
element, and count how many entries are less than or equal to that threshold.

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

Read `observed` from `surprise`, the tensor computed before truncation. If you
renormalize the kept probabilities and take the logarithm of those instead, the
update receives a smaller surprise than the specification requires.
