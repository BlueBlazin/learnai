from collections.abc import Callable


def time_region(fn: Callable[[], object], device_api, n: int) -> float:
    """Median seconds per call of `fn`, measured so that asynchronous work is counted.

    fn: a function of no arguments that queues work on a device and returns at once.
    device_api: an object with two methods. `device_api.sync()` waits until the device
        has finished everything queued (torch.cuda.synchronize); `device_api.clock()`
        returns the host time in seconds (time.perf_counter) and never waits.
    n: the number of timed calls, n >= 1.
    Calls `fn` exactly n + 1 times: one untimed warm-up call, then n timed calls.
    Returns the median of the n durations as a Python float.
    """
    raise NotImplementedError
