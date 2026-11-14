Start from what one step needs. The live beams are rows of a batch: row `i` has a
log-probability `S[i]` and next-token logits. The candidates are all pairs (row,
token), and their log-probabilities form a `(rows, V)` table:
`S[:, None] + log_softmax(logits)`. Flatten it and the index of a candidate is
`row * V + token`. After you have picked the next live beams, each of them remembers
which row it came from. What does the model's cache have to look like before you
can feed the new tokens as a batch?

---

Keep four things per step: the token lists of the live beams, their
log-probabilities, the cache (one row per live beam, in the same order), and the
finished list. One step:

1. sort the flattened table in descending order and take the first `2 * n_beams`
   indices; `divmod(index, V)` gives `(parent row, token)`;
2. walk them by rank: EOS (or any token at the last step) goes to the finished list
   if `rank < n_beams`, with score `total / step ** length_penalty`; anything else
   is appended to the new live list while it has fewer than `n_beams` entries;
3. sort the finished list by score (a stable sort) and cut it to `n_beams`;
4. return if this was the last step or the stop rule holds;
5. otherwise `cache = cache.select(parents)` and feed `[[token], ...]`.

The running value of a live beam is always the plain sum of log-probabilities. The
length penalty is applied once, when a hypothesis finishes, and in the stop rule.

---

The two lines that carry the problem:

```
logits, cache = lm([[token] for (parent, token, total) in live],
                   cache.select([parent for (parent, token, total) in live]))
```

`select` both expands (after step 1 all parents are row 0) and reorders. And the
stop rule, checked before that call:

```
if len(finished) == n_beams and live[0].total / max_new_tokens ** length_penalty <= finished[-1].score:
    return finished
```

`live[0]` is the best live beam because the live list is in rank order, and
`finished[-1]` is the worst finished hypothesis because the list is sorted. Start
with `logits, cache = lm(prompt)` and the single empty beam with total `0.0`; step 1
then needs no special case except that its table has one row.
