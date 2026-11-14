import math


def percentile_nearest_rank(values: list[float], q: float) -> float:
    """The q-th percentile by the nearest-rank method: always one of the observations.

    values: a non-empty list of floats in any order; must not be modified.
    q: the percentile, 0 <= q <= 100.
    Returns the element of rank max(1, ceil(q * n / 100)) in ascending order, where
    n = len(values) and ranks start at 1.
    """
    raise NotImplementedError


def percentile_linear(values: list[float], q: float) -> float:
    """The q-th percentile with linear interpolation (the default of numpy.percentile).

    values: a non-empty list of floats in any order; must not be modified.
    q: the percentile, 0 <= q <= 100.
    Returns a float: with s the sorted values (0-based) and h = (n - 1) * q / 100,
    the value s[floor(h)] + (h - floor(h)) * (s[floor(h) + 1] - s[floor(h)]).
    """
    raise NotImplementedError


def merge_percentile(shards: list[list[float]], q: float) -> float:
    """The nearest-rank q-th percentile of all samples of all shards together.

    shards: one list of raw samples per shard; a shard may be empty, but there is at
        least one sample in total. Must not be modified.
    Returns the same value as percentile_nearest_rank of the concatenated samples.
    """
    raise NotImplementedError
