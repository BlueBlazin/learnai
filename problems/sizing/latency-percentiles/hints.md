A percentile is a question about the sorted samples: "which value has q percent of
the data at or below it?" Start by sorting a copy. Then the two methods differ only
in how they turn `q` into a place in the sorted list: one picks a whole-numbered rank
and returns the sample there, the other computes a fractional position and blends
the two samples around it. For the shards: is there any way to recover the tenth
slowest request overall from each server's own p90?

---

Nearest rank: the rank is `q * n / 100` rounded **up**, and ranks start at 1 while
list indices start at 0; a rank of 0 (only when `q` is 0) becomes 1. Linear: the
position is `h = (n - 1) * q / 100` on a 0-based scale, so `q = 0` lands on the first
sample and `q = 100` on the last; split `h` into its whole part `i` and its
fractional part, and move that fraction of the way from `ordered[i]` to
`ordered[i + 1]`. Take care that `i + 1` does not run off the end when `q` is 100.
Merging: no arithmetic on per-shard results can work, because the answer depends on
how the shards' samples interleave. Put all samples into one list first.

---

```python
rank = max(1, math.ceil(q * n / 100))      # multiply before dividing
nearest = ordered[rank - 1]

h = (n - 1) * q / 100
i = math.floor(h)
linear = ordered[i] + (h - i) * (ordered[min(i + 1, n - 1)] - ordered[i])
```

and `merge_percentile` is `percentile_nearest_rank` applied to
`[v for shard in shards for v in shard]`. Use `sorted(values)`, not `values.sort()`:
the caller's list must stay as it was.
