# BUG: the 1-based rank is used directly as a 0-based index (clamped to the last element), so the result is one observation too high.
import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) / 100))
    return ordered[min(rank, len(ordered) - 1)]


def percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def merge_percentile(shards: list[list[float]], q: float) -> float:
    pooled = [value for shard in shards for value in shard]
    return percentile_nearest_rank(pooled, q)
