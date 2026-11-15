# BUG: reads the stop time first and synchronizes afterward; the wait happens, but outside the measured interval.
import statistics
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()
    samples = []
    for _ in range(n):
        device_api.sync()
        start = device_api.clock()
        fn()
        stop = device_api.clock()
        device_api.sync()
        samples.append(stop - start)
    return float(statistics.median(samples))
