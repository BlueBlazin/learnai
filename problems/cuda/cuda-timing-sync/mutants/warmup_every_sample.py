# BUG: makes an untimed call before every timed one; the result is right, and the benchmark runs fn twice as often as asked.
import statistics
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    samples = []
    for _ in range(n):
        fn()
        device_api.sync()
        start = device_api.clock()
        fn()
        device_api.sync()
        samples.append(device_api.clock() - start)
    return float(statistics.median(samples))
