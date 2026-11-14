## The idea

Sort the samples; a percentile is then a position in the sorted list. The two
standard rules differ in which positions exist.

**Nearest rank.** Only whole ranks exist. The $q$-th percentile is the smallest
observation with at least $q$ percent of the samples at or below it. With $n$
samples, "at least $q$ percent" means at least $qn/100$ of them, and the smallest
whole number of samples that is enough is $\lceil qn/100 \rceil$. So the answer is
the sample of that rank (rank 1 for $q = 0$, where the ceiling would be 0). The
result is always a latency that some request really had, and it never requires
arithmetic on the samples, so it is exactly reproducible.

**Linear interpolation.** Positions are continuous. The sorted samples are placed at
positions $0, 1, \dots, n-1$, percentile $q$ sits at $h = (n-1)\,q/100$, and the
value is read off the straight line between the two neighbors. This is method 7 of
Hyndman and Fan (1996) and the default of `numpy.percentile` and `pandas`
`quantile`. It varies smoothly with $q$, at the price of returning values that were
never observed. The scale is $n - 1$, not $n$, because there are $n - 1$ gaps
between $n$ samples: $q = 100$ must land on the last one.

**Merging.** A percentile is an order statistic of the *whole* sample. A shard's
percentile says where that rank falls among the shard's own samples and nothing
about how they interleave with everyone else's, so no function of the per-shard
percentiles (mean, weighted mean, maximum) gives the pooled percentile in general.
In the statement's example, the pooled p90 is the ninth of ten samples, which is the
*smaller* sample of the slow shard; the shard percentiles are 17 and 900, and 100 is
not recoverable from them. The only correct inputs are the raw samples, or a summary
designed to be merged.

## Reference solution (Python)

```python
import math


def percentile_nearest_rank(values, q):
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) / 100))
    return ordered[rank - 1]


def percentile_linear(values, q):
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def merge_percentile(shards, q):
    pooled = [value for shard in shards for value in shard]
    return percentile_nearest_rank(pooled, q)
```

## The floating-point trap in the rank

$\lceil x \rceil$ is discontinuous at whole numbers, so an error of one unit in the
last place of $x$ changes the rank by one exactly when $qn/100$ should be whole,
which is the common case (p99 of 100 samples, p99.9 of 1000). `q * n / 100` is safe
there: for an integer `q` the product is exact, and the single division rounds
correctly, so a whole result comes out whole. `q / 100 * n` rounds `q / 100` first
(0.55, 0.28, and 0.999 are not binary fractions) and multiplies the error by `n`:

```python
55 / 100 * 100      # 55.00000000000001   -> rank 56
55 * 100 / 100      # 55.0                -> rank 55
99.9 / 100 * 1000   # 999.0000000000001   -> rank 1000: "p99.9" silently becomes the maximum
99.9 * 1000 / 100   # 999.0               -> rank 999
```

The usual percentiles (50, 90, 95, 99) happen to survive the wrong order for every
sample size up to several thousand, which is how the bug stays hidden in code that is
only ever asked for p99. For arbitrary decimal `q` and very large `n`, exact
arithmetic (`fractions.Fraction`, or integer per-mille ranks) is the robust choice.

## Common mistakes

- **Averaging shard percentiles.** The mean of each shard's p99 is not a percentile
  of anything. It weights a server with ten requests like one with ten thousand, and
  it dilutes one bad replica's tail with the healthy ones.
- **Weighting the average by shard size.** It looks like the fix and is not:
  percentiles are not additive under any weighting. In the example it gives 193.6
  where the answer is 100.
- **Rounding the rank down.** `floor(q * n / 100)` returns a value with *fewer* than
  $q$ percent of the samples at or below it whenever $qn/100$ is not whole: p99 of
  ten samples becomes the ninth instead of the slowest.
- **Using the rank as an index.** Ranks are 1-based, lists 0-based. `ordered[rank]`
  is one observation too high and runs off the end at `q = 100`.
- **Dividing before multiplying** in the rank (above).
- **Assuming the input is sorted.** Latencies arrive in completion order. Indexing
  the unsorted list returns an arbitrary request's latency, and nothing fails.
- **Sorting the caller's list in place.** `values.sort()` gives right answers and
  destroys the arrival order that the caller may still need (for a time series, for
  pairing with request ids).
- **Mixing up the two definitions.** Returning the nearest-rank value from
  `percentile_linear`, or placing the interpolated percentile at `n * q / 100`
  instead of `(n - 1) * q / 100`. On large samples the methods nearly agree; on the
  ten-request example they give 130 and 347 for p90.

## Complexity

$O(n \log n)$ time for the sort and $O(n)$ memory for the sorted copy, per call.
Selection (`heapq.nlargest` for a high percentile, or quickselect) brings one
percentile down to $O(n)$; when several percentiles of the same data are needed,
sort once and index.

## In the wild

- **Know which definition a tool uses.** `numpy.percentile` interpolates linearly by
  default and offers nearest-rank variants through its `method` argument
  (`"inverted_cdf"` is the definition used here). Benchmark harnesses differ, so two
  reports of "p99" from the same run can differ, most visibly with few samples: with
  fewer than 100 samples, nearest-rank p99 is simply the maximum.
- **Mergeable summaries.** Keeping every sample is not always possible. Histograms
  with fixed bucket boundaries merge by adding counts, which is why Prometheus
  computes quantiles from summed bucket counters (`histogram_quantile` over a `sum`
  by bucket) and why averaging per-instance quantile gauges is a documented
  anti-pattern there. HdrHistogram, t-digest, and DDSketch are the same idea with
  bounded relative error. The result is an estimate whose error is set by the bucket
  width.
- **Sample size.** A p99 from 200 requests is decided by the three slowest; a p99.9 from
  fewer than 1000 samples is the maximum. Report the count next to the percentile.
