Implement nearest-rank percentiles, linearly interpolated percentiles, and exact percentile merging across sharded latency measurements.

## Specification

```python
def percentile_nearest_rank(values: list[float], q: float) -> float: ...
def percentile_linear(values: list[float], q: float) -> float: ...
def merge_percentile(shards: list[list[float]], q: float) -> float: ...
```

In the first two functions, `values` is a non-empty list of $n$ finite floats in any order, possibly containing duplicates. In all three functions, `q` is a percentile in $0 \le q \le 100$. Write $s_1 \le s_2 \le \dots \le s_n$ for the values sorted in ascending order using 1-based indexing.

**`percentile_nearest_rank(values, q)`** returns the observation at 1-based rank

$$
r = \max\!\big(1, \lceil q \cdot n / 100 \rceil\big),
$$

which is $s_r$. The result is always one of the input values, specifically the smallest observation such that at least $q$ percent of the samples are less than or equal to it. Passing `q = 0` returns the minimum and passing `q = 100` returns the maximum.

> [!WARNING]
> Compute the rank as `math.ceil(q * n / 100)` so the multiplication happens before the division. Computing `q / 100 * n` rounds `q / 100` first and can overshoot an integer by a tiny amount. For example, `99.9 / 100 * 1000` evaluates to `999.0000000000001` (ceiling `1000`), whereas `99.9 * 1000 / 100` is exactly `999.0`. The test suite includes inputs that trigger this difference.

**`percentile_linear(values, q)`** computes the percentile by linear interpolation between neighboring sorted observations, matching the default method of `numpy.percentile`. Using 0-based sorted values $x_0 \le \dots \le x_{n-1}$ (so $x_i = s_{i+1}$), compute the fractional index $h$, its integer floor $i$, and the interpolated value:

$$
h = (n - 1) \cdot \frac{q}{100}, \qquad i = \lfloor h \rfloor, \qquad
\text{result} = x_i + (h - i)\,(x_{i+1} - x_i).
$$

When $i = n - 1$, return $x_{n-1}$. The return value is a Python `float` and is generally not one of the original observations.

**`merge_percentile(shards, q)`** takes a list of sample lists (such as measurements from separate servers, load-generator workers, or time windows) and returns the nearest-rank percentile of all samples pooled together. That is, it returns `percentile_nearest_rank(pooled, q)`, where `pooled` contains every sample from every shard. Individual shards may be empty, and there is at least one sample in total. Neither the order of shards nor how samples are split across them affects the result.

None of the functions may modify its input arguments.

## Example

Consider ten request latencies in milliseconds, given in arrival order:

```python
latencies = [120.0, 95.0, 110.0, 2300.0, 105.0, 98.0, 130.0, 101.0, 99.0, 115.0]
# sorted:     95, 98, 99, 101, 105, 110, 115, 120, 130, 2300

percentile_nearest_rank(latencies, 50)   # 105.0    rank ceil(5.0) = 5
percentile_nearest_rank(latencies, 90)   # 130.0    rank ceil(9.0) = 9
percentile_nearest_rank(latencies, 99)   # 2300.0   rank ceil(9.9) = 10
percentile_nearest_rank(latencies, 0)    # 95.0     rank max(1, 0) = 1

percentile_linear(latencies, 50)         # 107.5    h = 4.5: halfway from 105 to 110
percentile_linear(latencies, 90)         # 347.0    h = 8.1: 130 + 0.1 * (2300 - 130)
percentile_linear(latencies, 99)         # 2104.7   h = 8.91: 130 + 0.91 * 2170
```

For the same ten requests, nearest-rank reports a p90 of 130 ms while linear interpolation reports 347 ms. Neither is wrong, though 347 ms was never observed because it lies on the line between the ninth and tenth values.

Now consider two servers, where one handled eight fast requests and the other handled two slow requests:

```python
a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
b = [100.0, 900.0]

percentile_nearest_rank(a, 90)   # 17.0
percentile_nearest_rank(b, 90)   # 900.0
merge_percentile([a, b], 90)     # 100.0   rank 9 of the ten pooled samples
```

The unweighted mean of the two shard p90s is 458.5, and the mean weighted by shard size is 193.6. The true p90 of all ten requests is 100.0.

## Constraints

- `1 <= n <= 100,000` samples in total per call, with finite floats in $[0, 10^6]$.
- `q` is an `int` or `float` with at most two decimal places (such as `50`, `99`, `99.9`, or `99.99`), with $0 \le q \le 100$. For every test input, `math.ceil(q * n / 100)` evaluates to the exact mathematical rank.
- `percentile_nearest_rank` and `merge_percentile` are compared exactly.
- `percentile_linear` is compared with a relative tolerance of `1e-9` and an absolute tolerance of `1e-9`.
- Sorting is expected. There is no speed requirement beyond the time limit.

## Why this matters

Latency SLOs are written as percentiles (such as p99 time to first token under 500 ms) because averages hide the slow tail requests that users notice. On small samples, nearest-rank and linear interpolation can differ by nearly the full gap between the two slowest requests, so two benchmark tools run on the same trace can disagree simply because they use different rules.

Incorrect aggregation is an even costlier mistake. A dashboard that averages each replica's p99 produces a number that is not the p99 of any population. Worse, it usually understates tail latency when it matters most, because one overloaded replica's tail gets diluted by healthy replicas. Global percentiles must be computed from pooled samples or from mergeable summaries such as histograms, never by combining precomputed percentiles.
