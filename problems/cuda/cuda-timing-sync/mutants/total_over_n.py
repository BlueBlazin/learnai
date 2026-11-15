# BUG: times all n calls as one region and divides by n, which is the mean per call, not the median of the per-call durations.
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()
    device_api.sync()
    start = device_api.clock()
    for _ in range(n):
        fn()
    device_api.sync()
    return (device_api.clock() - start) / n
