The router only needs one piece of state: for each replica `id`, store the last `index_size` prompts routed to it. Every routing decision follows from comparing match lengths computed from that history against `queue_len`. Start by writing a helper that returns the longest common prefix length of two token sequences.

---

Compute `least` first, then filter `replicas` to the eligible candidates before searching for the best match. A replica that exceeds the imbalance limit must not win on affinity no matter how long its prefix match is. By contrast, the low-match fallback considers all replicas in `replicas`. Finally, always append `tokens` to the chosen replica's record regardless of whether affinity or the fallback chose it.

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

`match` returns the maximum common prefix length across the replica's remembered prompts, or `0` if the record is empty. Using `collections.deque(maxlen=self.index_size)` automatically drops the oldest prompt when full.
