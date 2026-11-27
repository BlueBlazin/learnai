Top-k keeps a fixed number of tokens and top-p keeps a fixed amount of probability
mass. Min-p instead keeps a fixed *ratio*, retaining every token at least `min_p`
times as likely as the best one. What are the two quantities in that comparison, and
which one is computed per row so it must broadcast across the vocabulary axis?

---

Take the softmax along the last axis, then compute each row's maximum with the
reduced axis kept so it has shape `(B, 1)`. Multiply by `min_p` to get one cutoff
per row, and compare the probabilities against it using `>=` to form a `(B, V)`
boolean mask. Verify both boundary values: with `min_p = 1` the top token must still
pass, and with `min_p = 0` every token must pass.

---

```
probs = softmax(logits, axis=-1)
cutoff = min_p * max(probs, axis=-1, keepdims=True)
out = where(probs >= cutoff, logits, -inf)
```

Compute the mask from probabilities, then apply it to the *logits*, replacing
removed entries with `-inf` rather than `0`. If you prefer working in logit space,
the same condition is `logits >= max(logits) + log(min_p)` because a ratio of
probabilities is a difference of logits and the softmax normalizer cancels. Handle
`min_p = 0` separately in logit space to avoid evaluating `log(0)`.
