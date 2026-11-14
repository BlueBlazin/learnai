Think in token numbers, not in slots. After `end = offset + T` tokens have been
written, which token numbers does the cache hold? Write that list down for
`max_size = 5`, `keep = 2`, and `end = 4, 6, 8, 9` (the four writes of the worked
example), and then write the slot of each token next to it. The whole function is in
those two lists.

---

Two small pieces do all the work:

- the cached token numbers after the write, oldest first: `0 .. min(keep, end) - 1`
  followed by `max(keep, end - W) .. end - 1`, with `W = max_size - keep`;
- the slot of a token: `t` if `t < keep`, else `keep + (t - keep) % W`.

**Write:** each new token that is in the list goes to its slot. New tokens that are
not in the list (a chunk longer than the window) are skipped. **Read:** gather the
buffer at the slots of the list, in list order. That is temporal order by
construction, with no case analysis for "wrapped" or "not full yet".

---

With index tensors, `live` holding the cached token numbers and
`slots = where(live < keep, live, keep + (live - keep) % W)`:

```
fresh = number of entries of live that are >= offset      # they are its tail
k[:, :, slots[-fresh:]] = k_new[:, :, live[-fresh:] - offset]     # same for v
keys, values = k[:, :, slots], v[:, :, slots]
```

In JAX the assignment is `k = k.at[:, :, slots[-fresh:]].set(...)`. If you prefer
slices to index tensors: the ring part of a write is at most two contiguous pieces
(up to the end of the buffer, then from slot `keep`), and temporal order after a wrap
is `[sinks, ring from the oldest slot to the end, ring from keep up to the oldest
slot]`, where the oldest slot is `keep + (end - keep) % W`.
