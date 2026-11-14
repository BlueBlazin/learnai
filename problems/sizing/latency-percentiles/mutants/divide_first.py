# BUG: the rank is computed as ceil(q / 100 * n); q / 100 is rounded before the multiplication, so a product that should be a whole number can land just above it and the rank comes out one too high.
import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def merge_percentile(shards: list[list[float]], q: float) -> float:
    pooled = [value for shard in shards for value in shard]
    return percentile_nearest_rank(pooled, q)
