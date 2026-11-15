Measure how long a function that runs on a GPU takes per call. A CUDA call returns as
soon as its work is *queued*, so a timer around the call measures the queueing; the
host has to wait for the device at the right two moments, and the first call has to
be kept out of the numbers.

## Specification

```python
def time_region(fn, device_api, n): ...
```

- `fn`: a function of no arguments. Calling it queues work on a device and returns
  without waiting for that work.
- `device_api`: an object with two methods, standing for the two things a benchmark
  takes from the outside world:
  - `device_api.sync()` blocks until the device has finished everything queued so
    far. On a GPU it is `torch.cuda.synchronize()`.
  - `device_api.clock()` returns the host's time in seconds as a float, and never
    waits for the device. On a real machine it is `time.perf_counter()`.
- `n`: the number of timed calls, $n \ge 1$.
- Returns the median duration of one call, in seconds, as a Python `float`.

The measurement is defined as follows.

1. **Warm-up.** Call `fn()` once. This call is not timed.
2. **Samples.** Then call `fn()` exactly `n` more times. The duration of a call is
   the host time from *start* to *stop*, where
   - *start* is read with `device_api.clock()` at a moment when the device is idle:
     everything queued before this call (the warm-up, the previous call, work that
     was already running when `time_region` was called) has finished;
   - *stop* is read with `device_api.clock()` when the device has finished the work
     queued by this call.
3. **Result.** Return the median of the `n` durations: the middle one in sorted
   order for an odd `n`, the mean of the two middle ones for an even `n`.

`fn` is called exactly $n + 1$ times in total. All time is read from
`device_api.clock()`, never from another clock.

![One call on the host and on the device. A timer stopped when `fn()` returns measures the launch; a timer stopped after `sync()` measures the work.](assets/async-timeline.svg)

## Example

```python
dev = FakeDevice()  # from whetstone.kit
fn = dev.kernel(0.004, first_call_s=0.050)  # 4 ms of device work per call, 50 ms the first time

time_region(fn, dev, 5)  # 0.004
```

`FakeDevice` is a device on a virtual clock. `fn()` queues work and returns at host
time unchanged; `dev.sync()` moves the host's time forward to the moment the queue is
empty; `dev.clock()` reads the host's time. The run above, in host seconds:

| step | host time | device busy until |
|---|---|---|
| warm-up `fn()` | 0 | 0.050 |
| `sync()` | 0.050 | |
| `clock()` gives start = 0.050, then `fn()` | 0.050 | 0.054 |
| `sync()`, then `clock()` gives stop = 0.054 | 0.054 | |

The first duration is $0.054 - 0.050 = 0.004$, and so are the other four. Three ways
to get it wrong on the same input:

- Without the `sync()` before *stop*, the stop is read at 0.050: the duration is 0.
- Without the warm-up, the first duration is 0.050. The median of five hides it; with
  `n = 1` the result is 0.050.
- Without the `sync()` before *start*, the first timer starts at 0 while the warm-up
  is still running, and the first duration is 0.054.

## Constraints

- $1 \le n \le 32$. Durations are between 0.1 ms and 1 s.
- On the virtual clock results are exact: they are compared to a relative tolerance
  of `1e-9`.
- `fn` must be called exactly $n + 1$ times; the judge counts the calls.
- **What runs where.** The cases without a hardware mark run on the CPU with
  `FakeDevice` (and one with a real clock and a function that sleeps). One case runs
  only on a machine with an NVIDIA GPU: `fn` queues matrix products between two
  `torch.cuda.Event(enable_timing=True)` markers, `device_api` is the kit's
  `RealDevice("cuda")` (`torch.cuda.synchronize` and `time.perf_counter`), and the
  result must be at least half of the median time the events report. The GPU case
  was written from the PyTorch 2.15 documentation and could not be run when the
  problem was authored.

## Why this matters

Every latency number about a model on a GPU (time per token, time to first token,
tokens per second, a before-and-after for an optimization) comes from a timer on the
host. A timer stopped without a synchronization reports microseconds for a forward
pass that takes tens of milliseconds, and the missing time reappears in whatever
later line happens to wait for the device, usually a `.item()` or a `.cpu()` that is
then blamed for being slow. A timer started without one charges the region for its
predecessor's work. And the first call of anything on a GPU pays once for memory
allocation, kernel selection, and lazy initialization; left in the sample it makes a
3 ms operation look like 50. Benchmarks that get this wrong are the usual source of
speedups that disappear in production.
