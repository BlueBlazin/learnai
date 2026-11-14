# BUG: percentile_linear does not interpolate: it returns the nearest-rank percentile.
import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) / 100))
    return ordered[rank - 1]


def percentile_linear(values: list[float], q: float) -> float:
    return float(percentile_nearest_rank(values, q))


def merge_percentile(shards: list[list[float]], q: float) -> float:
    pooled = [value for shard in shards for value in shard]
    return percentile_nearest_rank(pooled, q)
