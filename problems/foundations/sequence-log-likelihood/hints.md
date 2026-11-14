Write down which logits score which token for a row of four tokens `a b c d`. The
model reads `a` and predicts `b`; reads `a b` and predicts `c`; reads `a b c` and
predicts `d`. Which position's logits are never used, and which token is never
scored? That tells you how to slice `logits` and `tokens` so that they line up.

---

Slice both tensors to `T - 1` columns: `logits[:, :-1]` (the contexts) and
`tokens[:, 1:]` (the targets). Take `log_softmax` over the vocabulary axis, never
`log(softmax(...))`, and pick out one entry per position with the target as the index
along the last axis (`gather` / `take_along_axis` with the targets given a trailing
axis of length 1). You now have a `(B, T - 1)` tensor of log-probabilities.

The mask needs the same treatment: pair `t` is made of positions `t` and `t + 1`, so
its mask is built from `mask[:, :-1]` and `mask[:, 1:]`.

---

```
counted = mask[:, :-1] & mask[:, 1:]              # (B, T - 1)
kept    = where(counted, picked, 0.0)             # not picked * counted
logprob = kept.sum(axis=1)
ppl     = exp(-logprob / counted.sum(axis=1))
```

Use `where` to drop the uncounted pairs: an uncounted log-probability can be `-inf`,
and `0 * -inf` is NaN. The perplexity divides by the number of counted pairs in the
row, which is neither `T - 1` nor the number of real tokens.
