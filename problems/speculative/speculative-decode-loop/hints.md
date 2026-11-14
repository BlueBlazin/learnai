Start with the bookkeeping, not the models. Keep one list `seq` (prompt plus
generated tokens) and, for each model, the number of tokens of `seq` its cache
covers. What does a model have to be fed, in terms of those two things, before its
last row of logits scores the token after `seq`? And after a round in which only
`n` of the `k` drafts survive, which of the tokens each cache covers are still part
of `seq`?

---

A model's next call always feeds `(seq + drafts)[covered:]`: everything it has not
seen. For the draft that is one call per proposal (the first call of a round
catches up on verified tokens, each later call feeds the previous proposal). For
the target it is a single call per round with the unseen verified tokens and all
drafts; its last `k' + 1` rows are the ones `verify_greedy` wants: row `j` scores
the position of draft `j`, and the last row is the bonus position. Compare, extend
`seq` by the accepted drafts and the target's token, check EOS and the budget.

---

Rollback is one rule for both models: after the round, a cache may cover at most
`len(seq) - 1` tokens, namely verified tokens only, and never the newest one,
which must be fed next round because its logits start that round.

```
t_len = len(seq) - 1;               t_cache = t_cache.truncate(t_len)
d_len = min(d_len, len(seq) - 1);   d_cache = d_cache.truncate(d_len)
```

The `min` matters after a fully accepted round: the draft never fed its own last
proposal, so its cache is already shorter than `len(seq) - 1`, and its next call
feeds two tokens. To respect the budget, draft only `min(k, remaining - 1)` tokens
per round, and cut the round's tokens after the first `eos_id`.
