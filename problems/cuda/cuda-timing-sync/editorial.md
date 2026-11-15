## The idea

PyTorch's CUDA notes say it in one sentence: "By default, GPU operations are
asynchronous." A call such as `x @ w` checks its arguments, allocates the result,
puts a kernel launch in the device's queue, and returns a tensor whose contents do
not exist yet. The host runs ahead; the device works through its queue in order. The
program still behaves as if everything were synchronous, because any operation that
needs a value on the host (copying to the CPU, `.item()`, printing) first waits for
the queue to drain.

A timer lives on the host, so what it measures depends on where the host waits:

- **Before the stop.** Without a wait, the stop is read when the launch returns, and
  the difference is the launch overhead. `torch.cuda.synchronize()` blocks until all
  queued kernels have completed; reading the clock after it gives the time at which
  the work was done.
- **Before the start.** The queue is first-in, first-out. If earlier work is still
  running when the timer starts, the timed work waits behind it, and the sample
  includes the remainder of that earlier work. Synchronizing before the start makes
  the device idle at the start.
- **The first call.** It pays one-time costs: memory for the allocator's pool,
  library handles, algorithm selection, sometimes compilation. A warm-up call that
  is not timed removes them from the samples.

The median is the right summary for the same practical reason: a single sample that
was hit by a context switch or a garbage collection does not move it.

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

The synchronization at the top of the loop is needed once, after the warm-up; in
later iterations the device is already idle and it returns immediately. Keeping it
inside the loop makes each sample correct on its own.

On a real GPU, `device_api.sync` is `torch.cuda.synchronize` and `device_api.clock`
is `time.perf_counter`. The kit's `FakeDevice` models the same two calls on a virtual
clock: `fn()` queues a duration and returns, `sync()` advances the host's time to
when the queue is empty, and `clock()` reads the host's time. That turns each mistake
into an exact wrong number on a machine without a GPU.

## Common mistakes

- **No synchronization before the stop.** The result is the launch time: 0 on the
  virtual clock, some microseconds on a GPU, for any amount of work.
- **Reading the stop time and then synchronizing.** The wait is there, on the wrong
  side of the clock read.
- **No synchronization before the start.** The first sample pays for the warm-up
  call and for whatever was queued earlier. With five samples the median hides it;
  with one or two it is the result.
- **No warm-up.** The slow first call is a sample.
- **The mean instead of the median**, directly or by timing all `n` calls as one
  region and dividing by `n`. Timing a whole loop between two synchronizations is a
  legitimate measurement of a mean; it is not the median this function is specified
  to return, and one outlier moves it.
- **The middle element without sorting**, or **the upper middle element** for an
  even count.
- **`time.perf_counter()` instead of `device_api.clock()`.** The function was handed
  its clock; a benchmark harness that injects one (to test, or to use device events)
  gets numbers from a clock it did not choose.
- **A warm-up before every sample.** The numbers are right and the benchmark costs
  twice as much.

## Complexity

$n + 1$ calls of `fn`, $2n$ synchronizations and clock reads, and a sort of `n`
numbers.

## In the wild

The CUDA notes give the two tools: `torch.cuda.synchronize()` before each clock read,
or a pair of `torch.cuda.Event(enable_timing=True)` objects recorded around the work,
whose `start.elapsed_time(end)` is the time in milliseconds between the two markers
on the device's own timeline. Events do not stall the host, which makes them the
better choice inside a running server; `torch.utils.benchmark.Timer` handles warm-up
and synchronization for microbenchmarks. Setting `CUDA_LAUNCH_BLOCKING=1` makes every
launch synchronous, which is for debugging and slows everything down. These
statements about real GPUs are taken from the PyTorch 2.15 documentation; the
virtual-clock cases were run, the GPU case was not.
