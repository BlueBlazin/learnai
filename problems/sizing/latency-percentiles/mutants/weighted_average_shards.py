# BUG: merge_percentile averages the per-shard percentiles weighted by shard size; weighting does not make percentiles additive, the raw samples must be pooled.
import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered) / 100))
    return ordered[rank - 1]


def percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def merge_percentile(shards: list[list[float]], q: float) -> float:
    total = sum(len(shard) for shard in shards)
    weighted = sum(len(shard) * percentile_nearest_rank(shard, q) for shard in shards if shard)
    return weighted / total
