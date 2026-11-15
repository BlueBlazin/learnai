# BUG: picks the middle sample in the order of measurement, without sorting; that is the median only when the durations happen to be ordered.
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
    middle = n // 2
    if n % 2:
        return float(samples[middle])
    return (samples[middle - 1] + samples[middle]) / 2
