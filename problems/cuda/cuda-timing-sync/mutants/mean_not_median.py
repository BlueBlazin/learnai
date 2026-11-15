# BUG: returns the mean of the samples instead of the median; one slow call shifts the result.
import statistics
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
    return float(statistics.fmean(samples))
