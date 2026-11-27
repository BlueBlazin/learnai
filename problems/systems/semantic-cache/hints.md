Think of `get` as a filter followed by a selection. Which entries are legally eligible to answer this request based on namespace and expiration time? Once you filter to eligible entries, how do you pick the winner, and what goes wrong if you pick the closest entry before filtering?

---

Store all entries in a single list ordered from least recently used to most recently used. Give each entry its creation timestamp (`stored_at`) and an incrementing insertion counter. In `get`, scan the list while skipping entries in other namespaces, expired entries (`now - stored_at >= ttl_s`), and entries with cosine similarity below `threshold`. Track the maximum of `(similarity, counter)` so ties go to the most recently stored entry. On a hit, move the winning entry to the end of the list without modifying `stored_at`.

---

`put` takes three steps in strict order:

```python
self.entries = [e for e in self.entries if now - e.stored_at < self.ttl_s]
if len(self.entries) >= self.capacity:
    self.entries.pop(0)  # least recently used
self.entries.append(new_entry)
```

Purging expired entries first ensures a full cache never evicts a live entry while dead ones remain. In `namespace_of`, remember that an absent `"temperature"` defaults to `1.0`, so check `request.get("temperature", 1.0) != 0`.
