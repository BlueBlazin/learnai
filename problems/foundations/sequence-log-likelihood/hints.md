Trace which logits score which token for a four-token sequence `a b c d`. The model reads `a` to predict `b`, reads `a b` to predict `c`, and reads `a b c` to predict `d`. Notice which position's logits are never used and which token is never scored. That gives you the slices that align `logits` and `tokens`.

---

Slice both tensors to `T - 1` positions along the sequence axis: `logits[:, :-1]` for the contexts and `tokens[:, 1:]` for the targets. Apply `log_softmax` over the vocabulary axis (never `log(softmax(...))`), then gather the single log-probability at each target index along the last axis using `gather` or `take_along_axis` with a trailing dimension of size 1. This produces a `(B, T - 1)` tensor `picked` of pair log-probabilities.

Build the pair mask the same way. Because pair `t` uses positions `t` and `t + 1`, its mask depends on both `mask[:, :-1]` and `mask[:, 1:]`.

---

```
counted = mask[:, :-1] & mask[:, 1:]              # (B, T - 1)
kept    = where(counted, picked, 0.0)             # not picked * counted
logprob = kept.sum(axis=1)
ppl     = exp(-logprob / counted.sum(axis=1))
```

Use `where` instead of multiplication to exclude uncounted pairs, because an uncounted log-probability can be `-inf` and `0 * -inf` is `NaN`. For perplexity, divide by the number of counted pairs in each row, not `T - 1` and not the number of real tokens.
