import statistics
from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    fn()  # warm-up: allocations, kernel selection, lazy initialization; not timed

    samples = []
    for _ in range(n):
        # Wait for everything queued so far (the warm-up, the previous call, other
        # people's work), so that the timer starts with an idle device.
        device_api.sync()
        start = device_api.clock()
        fn()  # returns as soon as the work is queued
        device_api.sync()  # wait for the work itself before stopping the timer
        samples.append(device_api.clock() - start)
    return float(statistics.median(samples))
