A lookup is a filter followed by a choice. Which entries may the cache return for
this request at all (think of the namespace and of time), and only then: which of
them is the best? What goes wrong if you choose first and filter afterwards?

---

Keep the entries in one list ordered by use, least recently used first, and give
every entry the time it was stored and a running number. `get` scans the list:
skip entries of another namespace, skip expired ones (`now - stored_at >= ttl_s`),
skip similarities below the threshold, and keep the maximum of
`(similarity, number)`, which prefers the newer of two equal matches. On a hit,
move the entry to the end of the list. Do not touch `stored_at`.

---

`put` is three lines in a fixed order:

```python
self.entries = [e for e in self.entries if now - e.stored_at < self.ttl_s]
if len(self.entries) >= self.capacity:
    self.entries.pop(0)  # least recently used
self.entries.append(new_entry)
```

Dropping the expired entries first matters: otherwise a full cache evicts a live
entry while dead ones keep their places. For `namespace_of`, remember that a
missing temperature means 1.0: `request.get("temperature", 1.0) != 0`.
