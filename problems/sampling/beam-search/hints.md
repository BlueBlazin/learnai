Start with what one decoding step needs. Treat the live beams as rows of a batch, where row `i` has cumulative log-probability `S[i]` and next-token logits over vocabulary size `V`. All `(row, token)` extensions form a `(rows, V)` table `S[:, None] + log_softmax(logits)`. Flattening this table gives each candidate the index `row * V + token`. After choosing the next live beams, each beam knows which parent row it came from. How must the KV cache be updated before feeding the new tokens as a batch?

---

Track four values across steps: the live beam token lists, their cumulative log-probabilities, the KV cache (one row per live beam, in the same order), and the finished list. At each step:

1. Sort the flattened candidate table in descending order, take the first `2 * n_beams` indices, and unpack `(parent, token) = divmod(index, V)`.
2. Walk the candidates in rank order. If `token == eos_id` (or `step == max_new_tokens`), append the hypothesis to `finished` with score `total / step ** length_penalty` only when `rank < n_beams`. Otherwise, append it to the new live list until `len(live) == n_beams`.
3. Sort `finished` by score descending (using a stable sort) and keep only the first `n_beams` entries.
4. Stop and return if `step == max_new_tokens` or if the early-stopping bound holds.
5. Otherwise, reorder the cache with `cache = cache.select(parents)` and run `lm([[token], ...], cache)`.

Keep each live beam's running score as the plain sum of log-probabilities. Apply `length_penalty` only when scoring a finished hypothesis and when evaluating the stop bound.

---

These two lines handle cache expansion, reordering, and batched decoding:

```
logits, cache = lm([[token] for (parent, token, total) in live],
                   cache.select([parent for (parent, token, total) in live]))
```

`select` both expands the single prompt row (after step 1, every parent is row `0`) and reorders rows on later steps. Check the stop rule right before that model call:

```
if len(finished) == n_beams and live[0].total / max_new_tokens ** length_penalty <= finished[-1].score:
    return finished
```

`live[0]` is the highest-scoring live beam because `live` is built in rank order, and `finished[-1]` is the worst finished hypothesis because `finished` is sorted descending. Initialize with `logits, cache = lm(prompt)` and a single empty beam with total `0.0` so step 1 uses the exact same loop as every other step.
