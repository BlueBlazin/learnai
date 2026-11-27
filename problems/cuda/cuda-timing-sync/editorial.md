## The idea

GPU operations in PyTorch are asynchronous by default. A call such as `x @ w` validates its arguments, allocates the output buffer, enqueues a kernel launch on the device, and returns a tensor whose values have not been computed yet. The host keeps executing while the device works through its queue in order. Programs still see consistent results because any operation that needs a value on the host (copying to the CPU, `.item()`, or printing) blocks until the queue drains.

Because the timer runs on the host, what it measures depends on when the host waits for the device:

- **Sync before stopping.** Without a wait, the stop timestamp is read as soon as the kernel launch returns, measuring only launch overhead. Calling `torch.cuda.synchronize()` blocks until all queued kernels finish, so reading the clock afterward captures the actual device work.
- **Sync before starting.** The device queue is first-in, first-out. If earlier work is still running when the timer starts, the timed call waits behind it and inherits the remaining duration of that earlier work. Synchronizing before reading the start timestamp ensures the device is idle.
- **Warm-up call.** The first call to a GPU operation pays one-time costs such as memory pool allocation, library handle setup, algorithm selection, and compilation. Running one untimed warm-up call keeps those startup costs out of the samples.

Using the median instead of the mean protects the result against a single outlier caused by a host context switch or garbage collection pause.

## Reference solution (PyTorch)

```python
def time_region(fn, device_api, n):
    fn()  # warm-up

    samples = []
    for _ in range(n):
        device_api.sync()
        start = device_api.clock()
        fn()
        device_api.sync()
        samples.append(device_api.clock() - start)
    return float(statistics.median(samples))
```

The synchronization at the top of the loop is strictly needed on the first iteration after the warm-up. On later iterations, the device is already idle from the previous stop synchronization, so the call returns immediately. Keeping it inside the loop makes every iteration self-contained.

On a real GPU, `device_api.sync` is `torch.cuda.synchronize` and `device_api.clock` is `time.perf_counter`. The test harness's `FakeDevice` models both methods on a virtual clock so each timing mistake produces an exact wrong number even on a CPU-only machine.

## Common mistakes

- **No synchronization before stopping.** The timer measures only launch overhead, which is `0` on the virtual clock and a few microseconds on a GPU regardless of kernel size.
- **Reading the stop time before synchronizing.** The wait happens on the wrong side of the clock read, so the duration still excludes the device work.
- **No synchronization before starting.** The first sample absorbs the warm-up call and any previously queued work. Five samples hide this behind the median, but `n = 1` or `n = 2` exposes it.
- **No warm-up call.** The slow first call is included as a timed sample.
- **Returning the mean instead of the median.** This includes timing all `n` calls in one block between two synchronizations and dividing by `n`. That measures a mean rather than the specified median and is sensitive to single outliers.
- **Taking the middle element without sorting, or taking the upper middle element for an even `n`.** The median requires sorting and averaging the two middle values when `n` is even.
- **Calling `time.perf_counter()` directly instead of `device_api.clock()`.** Ignoring the injected clock breaks virtual-clock tests and custom timing harnesses.
- **Running a warm-up before every sample.** This doubles the number of `fn()` calls and violates the $n + 1$ call-count requirement.

## Complexity

`time_region` makes $n + 1$ calls to `fn`, $2n$ synchronizations and clock reads, and takes $O(n \log n)$ time to sort `n` durations.

## In the wild

PyTorch's CUDA documentation recommends two timing patterns. The first is calling `torch.cuda.synchronize()` before each host clock read. The second is recording a pair of `torch.cuda.Event(enable_timing=True)` markers around the work and calling `start.elapsed_time(end)` to read elapsed milliseconds on the device timeline. CUDA events avoid stalling the host, which makes them preferable inside a live server. For standalone microbenchmarks, `torch.utils.benchmark.Timer` handles warm-up and synchronization automatically. Setting `CUDA_LAUNCH_BLOCKING=1` forces every launch to be synchronous, which is useful for debugging errors but slows down execution. These real-GPU notes follow the PyTorch 2.15 documentation. The virtual-clock test cases were executed, while the GPU test case was not.
