Top-k keeps a fixed number of tokens and top-p a fixed amount of probability. Min-p
keeps a fixed *ratio*: every token at least `min_p` times as likely as the best one.
What are the two ingredients of that test, and which of them is a per-row quantity
that has to broadcast against the whole row?

---

Take the softmax along the last axis, then the maximum of each row with the reduced
axis kept, so it has shape `(B, 1)`. Multiply it by `min_p` to get one cutoff per row,
and compare the probabilities with it using `>=`. The comparison gives a `(B, V)`
boolean table of the tokens to keep. Check the two ends: with `min_p = 1` the top
token must still pass, and with `min_p = 0` everything must.

---

```
probs = softmax(logits, axis=-1)
cutoff = min_p * max(probs, axis=-1, keepdims=True)
out = where(probs >= cutoff, logits, -inf)
```

The mask is computed from probabilities but applied to the *logits*, and a removed
token becomes `-inf`, not 0. If you prefer to stay in logit space, the same test is
`logits >= max(logits) + log(min_p)`: a ratio of probabilities is a difference of
logits, and the softmax normalizer cancels. Handle `min_p = 0` separately there, since
`log(0)` is not something to compute.
