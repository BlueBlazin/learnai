import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)  # a new list: the caller's samples stay in arrival order
    # Multiply before dividing: q / 100 * n can land just above a whole number.
    rank = max(1, math.ceil(q * len(ordered) / 100))
    return ordered[rank - 1]  # ranks are 1-based


def percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100  # fractional 0-based position in the sorted list
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def merge_percentile(shards: list[list[float]], q: float) -> float:
    # Percentiles do not average. Pool the raw samples, then take one percentile.
    pooled = [value for shard in shards for value in shard]
    return percentile_nearest_rank(pooled, q)
