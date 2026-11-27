Start with the cache bookkeeping rather than the models. Keep one list `seq` (the prompt plus verified generated tokens) and, for each model, the number of tokens of `seq` its cache currently covers. Ask two questions: which slice of `seq` must you feed a model so its last logit row scores the token after `seq`, and after a round where only `n` of `k` drafts survive, how many cached tokens are still part of `seq`?

---

Each model call always feeds `(seq + drafts)[covered:]`, where `covered` is that model's cached token count, so each model sees only tokens it has not processed yet. The draft model runs once per proposal. The first call of a round catches up on unseen verified tokens, and each later call feeds the previous proposal. The target model runs once per round on all unseen verified tokens plus all drafts. Its last `k' + 1` logit rows give the target's greedy tokens for `verify_greedy`, where row `j` corresponds to draft `j` and the last row is the bonus position. Compare drafts against target tokens, append the accepted drafts and the target's token to `seq`, and check for `eos_id` and `max_new_tokens`.

---

Both caches follow one rollback rule at the end of a round. Keep at most `len(seq) - 1` tokens so that only verified tokens remain and the newest emitted token stays uncached, ready to be fed at the start of the next round:

```
t_len = len(seq) - 1;               t_cache = t_cache.truncate(t_len)
d_len = min(d_len, len(seq) - 1);   d_cache = d_cache.truncate(d_len)
```

The `min` handles fully accepted rounds, because the draft model never fed its last proposal and its cache is already one token shorter than `len(seq) - 1` (so the next round feeds two tokens). To stay within the token budget, draft `min(k, remaining - 1)` tokens per round, and slice the round's new tokens right after the first `eos_id`.
