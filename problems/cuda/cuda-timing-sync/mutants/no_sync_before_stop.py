# BUG: reads the stop time right after fn() returns, without synchronizing; the work is still queued, so only the time to launch it is measured.
import statistics
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()
    samples = []
    for _ in range(n):
        device_api.sync()
        start = device_api.clock()
        fn()
        samples.append(device_api.clock() - start)
    return float(statistics.median(samples))
