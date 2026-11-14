# BUG: percentile_linear places the percentile at position n * q / 100 instead of (n - 1) * q / 100, clamped to the last element.
import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) / 100))
    return ordered[rank - 1]


def percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = min(len(ordered) * q / 100, len(ordered) - 1)
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def merge_percentile(shards: list[list[float]], q: float) -> float:
    pooled = [value for shard in shards for value in shard]
    return percentile_nearest_rank(pooled, q)
