The router needs one data structure: per replica id, the last few prompts it sent
there. Everything else is a comparison of numbers you can compute from that record
and from `queue_len`. Start with the function that measures how many leading tokens
two prompts share.

---

Compute `least` first, then filter the replicas to the eligible ones before looking
for a match: a replica that is too busy must not win however well it matches. The
fallback, in contrast, looks at all replicas. Do not forget the last step: the
chosen replica is about to hold this prompt, so the record must say so, whichever
branch chose it.

---

```python
least = min(r.queue_len for r in replicas)
eligible = [r for r in replicas if r.queue_len - least <= self.imbalance]
best = max(eligible, key=lambda r: (self.match(tokens, r.id), -r.queue_len, -r.id))
if self.match(tokens, best.id) >= self.match_threshold * len(tokens):
    chosen = best
else:
    chosen = min(replicas, key=lambda r: (r.queue_len, r.id))
self.index.setdefault(chosen.id, deque(maxlen=self.index_size)).append(tuple(tokens))
return chosen.id
```

`match` is the maximum over the replica's remembered prompts of the common prefix
length, 0 if there are none. A `collections.deque(maxlen=n)` drops its oldest entry
by itself.
