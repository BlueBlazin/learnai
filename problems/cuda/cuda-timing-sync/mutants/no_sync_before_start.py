# BUG: synchronizes before stopping the timer but not before starting it; the first sample also pays for the warm-up call and for whatever was queued earlier.
import statistics
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()
    samples = []
    for _ in range(n):
        start = device_api.clock()
        fn()
        device_api.sync()
        samples.append(device_api.clock() - start)
    return float(statistics.median(samples))
