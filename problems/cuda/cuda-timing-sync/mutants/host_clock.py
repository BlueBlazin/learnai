# BUG: synchronizes through device_api but reads time.perf_counter() instead of device_api.clock(); the measurement is on a different clock than the one it was given.
import statistics
import time
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()
    samples = []
    for _ in range(n):
        device_api.sync()
        start = time.perf_counter()
        fn()
        device_api.sync()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))
