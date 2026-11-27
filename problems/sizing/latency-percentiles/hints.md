A percentile asks which value has `q` percent of the sorted data at or below it. Start by sorting a copy of the list. The two methods then differ only in how they turn `q` into a position. Nearest-rank picks a whole-numbered rank, while linear interpolation computes a fractional index and blends the two neighboring samples. For sharded data, think about whether you can recover the overall ninth-smallest value from each server's own p90.

---

For nearest rank, round `q * n / 100` **up** to get a 1-based rank, clamp rank 0 (when `q == 0`) to 1, and subtract 1 for 0-based indexing. For linear interpolation, compute the 0-based position `h = (n - 1) * q / 100` so `q = 0` maps to index `0` and `q = 100` maps to index `n - 1`. Split `h` into its integer floor `i` and fractional part `h - i`, then move that fraction of the way from `ordered[i]` to `ordered[i + 1]`, clamping `i + 1` at `n - 1` when `q == 100`. For merging, no formula on per-shard percentiles works because the result depends on how values interleave across shards, so pool all samples into one list first.

---

```python
rank = max(1, math.ceil(q * n / 100))      # multiply before dividing
nearest = ordered[rank - 1]

h = (n - 1) * q / 100
i = math.floor(h)
linear = ordered[i] + (h - i) * (ordered[min(i + 1, n - 1)] - ordered[i])
```

Implement `merge_percentile` by calling `percentile_nearest_rank` on `[v for shard in shards for v in shard]`. Use `sorted(values)` instead of `values.sort()` so the caller's list is not modified.
