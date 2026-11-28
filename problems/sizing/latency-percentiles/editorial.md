## The idea

Start by sorting the samples. Once the data is sorted, a percentile is simply a position in the ordered list, and the two definitions differ in how they map $q$ to that position.

**Nearest rank.** Only whole ranks exist. The $q$-th percentile is the smallest observation with at least $q$ percent of the samples at or below it. With $n$ samples, that requires at least $qn / 100$ samples, so the smallest sufficient whole rank is $\lceil qn / 100 \rceil$ (clamped to rank 1 when $q = 0$). Because it picks an existing observation without arithmetic on the values, the result is always an actual measured latency and is exactly reproducible.

**Linear interpolation.** Positions are continuous on the 0-based interval $[0, n - 1]$. Percentile $q$ maps to $h = (n - 1) \cdot q / 100$, and the result is read from the line segment connecting the two neighboring sorted samples. This is method 7 of Hyndman and Fan (1996) and the default in `numpy.percentile` and `pandas` `quantile`. The scale factor is $n - 1$ rather than $n$ because there are $n - 1$ gaps between $n$ samples, ensuring that $q = 100$ lands on the last element.

**Merging.** A percentile is an order statistic of the entire dataset. A shard's percentile tells you where a rank falls inside that shard alone, not how its values interleave with other shards. No function of per-shard percentiles (mean, weighted mean, or maximum) can recover the pooled percentile in general. In the worked example, the pooled p90 is the ninth of ten samples, which is 100.0 (the smaller sample of the slow shard), and cannot be recovered from the shard p90s of 17.0 and 900.0. Computing the exact global percentile requires pooling the raw samples first.

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

Because $\lceil x \rceil$ jumps at every integer, a rounding error of one unit in the last place (ULP) above a whole number shifts the rank up by one. That happens precisely when $qn / 100$ is mathematically an integer, which is common in practice (such as p99 of 100 samples or p99.9 of 1000).

Writing `q * n / 100` avoids this error for integer `q` and standard decimal percentiles. The product is exact or clean, and a single division rounds accurately so exact integers stay exact. Writing `q / 100 * n` rounds `q / 100` first (since 0.55, 0.28, and 0.999 have no exact binary representation) and scales the rounding error by `n`:

```python
55 / 100 * 100      # 55.00000000000001   -> rank 56
55 * 100 / 100      # 55.0                -> rank 55
99.9 / 100 * 1000   # 999.0000000000001   -> rank 1000: "p99.9" silently becomes the maximum
99.9 * 1000 / 100   # 999.0               -> rank 999
```

Common percentiles like 50, 90, 95, and 99 happen to avoid this bug for sample sizes up to several thousand, which lets it hide in code that is only tested on p99. For arbitrary decimal `q` and very large `n`, use exact arithmetic such as `fractions.Fraction` or integer per-mille ranks.

## Common mistakes

- **Averaging shard percentiles.** The mean of per-shard p99s is not a valid percentile of the combined data. It gives a server with 10 requests the same weight as one with 10,000, and it hides a bad replica's tail among healthy ones.
- **Weighting the average by shard size.** Percentiles are not additive under any weighting. In the worked example, the weighted average gives 193.6 instead of 100.0.
- **Rounding the rank down (`floor(q * n / 100)`).** Whenever $qn / 100$ is not an integer, flooring returns a value with fewer than $q$ percent of samples at or below it. On ten samples, p99 becomes the ninth value instead of the tenth.
- **Using the 1-based rank directly as a 0-based index.** `ordered[rank]` is off by one and raises `IndexError` when `q = 100`.
- **Dividing before multiplying (`q / 100 * n`).** This triggers the floating-point ceiling jump shown above.
- **Assuming the input is already sorted.** Latencies arrive in completion order, so indexing without sorting silently returns an arbitrary sample.
- **Sorting in place with `values.sort()`.** This mutates the caller's list and destroys arrival order needed for time series or request-ID pairing.
- **Mixing up the two definitions.** Returning nearest-rank from `percentile_linear`, or computing `h = n * q / 100` instead of `(n - 1) * q / 100`, produces wrong answers. The difference is small on huge datasets but large on small ones (130 ms vs. 347 ms for p90 in the worked example).

## Complexity

Each call takes $O(n \log n)$ time for sorting and $O(n)$ memory for the sorted copy. For a single percentile, selection algorithms such as quickselect or `heapq.nlargest` reduce time complexity to $O(n)$. When computing several percentiles from the same dataset, sort once and index multiple times.

## In the wild

- **Tool definitions vary.** `numpy.percentile` uses linear interpolation by default and exposes nearest-rank as `method="inverted_cdf"`. Different benchmark harnesses choose different defaults, so two tools reporting "p99" on the same run can disagree, especially on small samples where nearest-rank p99 of fewer than 100 samples is simply the maximum.
- **Mergeable summaries.** Storing every raw latency sample across a fleet is often impractical. Histograms with fixed bucket boundaries merge cleanly by summing bucket counts, which is why Prometheus computes quantiles with `histogram_quantile` over summed counters and warns against averaging per-instance quantile gauges. HdrHistogram, t-digest, and DDSketch apply the same mergeable design with bounded relative error determined by their bucket width.
- **Sample size matters.** A p99 computed from 200 requests is determined by the top three observations, and a p99.9 from fewer than 1,000 requests is just the maximum. Always report the sample count alongside tail percentiles.
