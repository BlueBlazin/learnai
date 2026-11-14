Implement the two percentile definitions that latency reports use, nearest-rank and
linear interpolation, and the percentile of measurements that arrive split across
shards. Each is a few lines; the substance is that "p99" is not one number until the
rule is stated, and that percentiles of parts cannot be combined into the percentile
of the whole.

## Specification

```python
def percentile_nearest_rank(values: list[float], q: float) -> float: ...
def percentile_linear(values: list[float], q: float) -> float: ...
def merge_percentile(shards: list[list[float]], q: float) -> float: ...
```

Throughout, `values` is a non-empty list of $n$ finite floats in **any order**,
possibly with duplicates, and `q` is the percentile, $0 \le q \le 100$. Write
$s_1 \le s_2 \le \dots \le s_n$ for the values in ascending order (1-based).

**`percentile_nearest_rank(values, q)`** returns the observation of rank

$$
r = \max\!\big(1, \lceil q \cdot n / 100 \rceil\big),
$$

that is, $s_r$. The result is always one of the values: the smallest observation
such that at least $q$ percent of the samples are less than or equal to it. `q = 0`
gives the minimum and `q = 100` the maximum.

> [!WARNING]
> Compute the rank as `math.ceil(q * n / 100)`, multiplying first. `q / 100 * n`
> rounds `q / 100` before the multiplication and can land just above a whole number:
> `99.9 / 100 * 1000` is `999.0000000000001`, whose ceiling is 1000, while
> `99.9 * 1000 / 100` is exactly `999.0`. The tests include such inputs.

**`percentile_linear(values, q)`** returns the percentile with linear interpolation
between neighboring observations, the default method of `numpy.percentile`. With
0-based sorted values $x_0 \le \dots \le x_{n-1}$ (so $x_i = s_{i+1}$):

$$
h = (n - 1) \cdot \frac{q}{100}, \qquad i = \lfloor h \rfloor, \qquad
\text{result} = x_i + (h - i)\,(x_{i+1} - x_i),
$$

and the result is $x_{n-1}$ when $i = n - 1$. It returns a float, which in general is
not one of the observations.

**`merge_percentile(shards, q)`** takes one list of raw samples per shard (per
server, per load-generator worker, per time window) and returns the nearest-rank
percentile of **all samples together**: exactly
`percentile_nearest_rank(pooled, q)`, where `pooled` contains every sample of every
shard. Individual shards may be empty; there is at least one sample in total. How the
samples are distributed over the shards, and the order of the shards, do not affect
the result.

None of the functions modifies its arguments.

## Example

Ten request latencies in milliseconds, in arrival order:

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

The two definitions report a p90 of 130 ms and 347 ms for the same ten requests.
Neither is wrong, but 347 ms is a latency no request had: it is a point on the line
between the ninth and tenth observations.

Two servers, one that handled eight fast requests and one that handled two slow
ones:

```python
a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
b = [100.0, 900.0]

percentile_nearest_rank(a, 90)   # 17.0
percentile_nearest_rank(b, 90)   # 900.0
merge_percentile([a, b], 90)     # 100.0   rank 9 of the ten pooled samples
```

The mean of the two shard values is 458.5 and the mean weighted by shard size is
193.6. The p90 of the ten requests is 100.0.

## Constraints

- `1 <= n <= 100,000` samples in total per call; values are finite and lie in
  $[0, 10^6]$.
- `q` is an int or a float with at most two decimal places (`50`, `99`, `99.9`,
  `99.99`), $0 \le q \le 100$. For every input in the tests,
  `math.ceil(q * n / 100)` evaluates to the mathematically exact rank.
- `percentile_nearest_rank` and `merge_percentile` are compared exactly.
- `percentile_linear` is compared with a relative tolerance of `1e-9` and an absolute
  tolerance of `1e-9`.
- Sorting is expected; there is no speed requirement beyond the time limit.

## Why this matters

Latency objectives are written as percentiles ("p99 time to first token under 500
ms") because the mean hides the slow requests that users notice. Whether a system
meets such an objective can depend on the definition: on a small sample, nearest-rank
and interpolated p99 differ by as much as the gap between the two slowest requests,
and two benchmark tools run on the same data will disagree for that reason alone.
Aggregation is the costlier mistake. A dashboard that averages each replica's p99
produces a number that is not the p99 of anything, and it is usually too low when it
matters most, because one overloaded replica's tail is averaged with healthy ones.
Percentiles must be computed from pooled samples, or from mergeable summaries such as
histograms, and never from other percentiles.
