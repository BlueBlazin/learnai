Think in terms of token numbers rather than buffer slots. After `end = offset + T` tokens have been written, which token numbers should remain in the cache? Write out that ordered list for `max_size = 5`, `keep = 2`, and `end = 4, 6, 8, 9` (the four steps in the example), then write each token's slot next to it.

---

Two formulas cover every case:

- **Live tokens (oldest first):** `0 .. min(keep, end) - 1`, followed by `max(keep, end - W) .. end - 1`, where `W = max_size - keep`.
- **Slot of token `t`:** `t` if `t < keep`, otherwise `keep + (t - keep) % W`.

To write, scatter every new token that appears in the live list into its slot, skipping any earlier tokens from an overlong chunk that already fell out of the window. To read, gather the buffer at the slots of the live list in order. This produces temporal order directly without branching on whether the buffer is full or wrapped.

---

With index arrays where `live` holds the cached token numbers and `slots = where(live < keep, live, keep + (live - keep) % W)`:

```python
fresh = number of entries in live that are >= offset  # the tail of live
k[:, :, slots[-fresh:]] = k_new[:, :, live[-fresh:] - offset]  # same for v
keys, values = k[:, :, slots], v[:, :, slots]
```

In JAX, replace the in-place assignment with `k = k.at[:, :, slots[-fresh:]].set(...)`. If you prefer slicing over index arrays, the ring part of a write spans at most two contiguous slices (up to `max_size`, then from `keep`), and temporal order after a wrap concatenates the buffer along axis 2 as `[buf[:keep], buf[oldest:], buf[keep:oldest]]`, where `oldest = keep + (end - keep) % W`.
