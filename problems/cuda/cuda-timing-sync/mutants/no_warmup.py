# BUG: has no warm-up call; the first, slow call is one of the n timed samples.
import statistics
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    samples = []
    for _ in range(n):
        device_api.sync()
        start = device_api.clock()
        fn()
        device_api.sync()
        samples.append(device_api.clock() - start)
    return float(statistics.median(samples))
