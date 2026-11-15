# BUG: takes the element at index n // 2 of the sorted samples; for an even n that is the upper of the two middle values, not their mean.
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()
    samples = []
    for _ in range(n):
        device_api.sync()
        start = device_api.clock()
        fn()
        device_api.sync()
        samples.append(device_api.clock() - start)
    return float(sorted(samples)[n // 2])
