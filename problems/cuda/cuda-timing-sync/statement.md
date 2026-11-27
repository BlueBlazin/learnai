Implement `time_region` to measure the per-call execution time of an asynchronous GPU function. Because CUDA calls return as soon as their work is queued on the device, a host timer must wait for the device to finish before starting and before stopping, and it must exclude the first warm-up call.

## Specification

```python
def time_region(fn, device_api, n): ...
```

- `fn`: a function of no arguments that queues work on a device and returns immediately without waiting for that work to finish.
- `device_api`: an object providing two methods:
  - `device_api.sync()` blocks the host until the device finishes all work queued so far. On a GPU, this corresponds to `torch.cuda.synchronize()`.
  - `device_api.clock()` returns the current host time in seconds as a `float` without waiting for the device. On a real machine, this corresponds to `time.perf_counter()`.
- `n`: the number of timed calls ($n \ge 1$).
- Returns the median duration of one call, in seconds, as a Python `float`.

Perform the measurement in three steps:

1. **Warm-up.** Call `fn()` once without timing it.
2. **Timed samples.** Call `fn()` exactly `n` more times. Measure each call as `stop - start`, where:
   - `start` is read from `device_api.clock()` only after the device is idle. All work queued before this call (the warm-up call, the previous timed call, or any work already running when `time_region` was entered) must have finished.
   - `stop` is read from `device_api.clock()` only after the device finishes the work queued by this call.
3. **Median.** Sort the `n` durations. Return the middle value if `n` is odd, or the mean of the two middle values if `n` is even.

Call `fn` exactly $n + 1$ times in total. Read all timestamps from `device_api.clock()`, never from another clock.

![One call on the host and on the device. A timer stopped when `fn()` returns measures the launch; a timer stopped after `sync()` measures the work.](assets/async-timeline.svg)

## Example

```python
dev = FakeDevice()  # from learnai.kit
fn = dev.kernel(0.004, first_call_s=0.050)  # 4 ms of device work per call, 50 ms the first time

time_region(fn, dev, 5)  # 0.004
```

`FakeDevice` simulates an asynchronous device on a virtual clock. Calling `fn()` queues work without advancing host time. Calling `dev.sync()` advances host time to the moment the device queue becomes empty. Calling `dev.clock()` returns the current host time. The trace below shows the first timed call in host seconds:

| step | host time | device busy until |
|---|---|---|
| warm-up `fn()` | 0 | 0.050 |
| `sync()` | 0.050 | |
| `clock()` gives start = 0.050, then `fn()` | 0.050 | 0.054 |
| `sync()`, then `clock()` gives stop = 0.054 | 0.054 | |

The first timed duration is $0.054 - 0.050 = 0.004$, and the remaining four samples are also `0.004`. Common mistakes produce wrong numbers on this same input:

- Omitting `sync()` before `stop` reads `0.050` right after `fn()` returns, giving a duration of `0`.
- Omitting the warm-up call makes the first sample `0.050`. The median of five samples hides this bug, but `n = 1` returns `0.050`.
- Omitting `sync()` before `start` starts the first timer at `0` while the warm-up is still running, giving a first sample of `0.054`.

## Constraints

- $1 \le n \le 32$, with durations between 0.1 ms and 1 s.
- Virtual-clock test cases are exact and compared to a relative tolerance of `1e-9`.
- `fn` must be called exactly $n + 1$ times. The judge counts every call.
- **What runs where.** Non-GPU test cases run on the CPU with `FakeDevice` (plus one case using a real clock and a sleeping function). One GPU test case runs only on a machine with an NVIDIA GPU. In that case, `fn` queues matrix products between two `torch.cuda.Event(enable_timing=True)` markers, `device_api` is `RealDevice("cuda")` (`torch.cuda.synchronize` and `time.perf_counter`), and the result must be at least half of the median time reported by the events. The GPU case was written from the PyTorch 2.15 documentation and was not run when the problem was authored.

## Why this matters

Every GPU latency metric (time per token, time to first token, tokens per second, and before-and-after optimization numbers) comes from a timer on the host. Stopping a timer without synchronizing reports microseconds for a forward pass that takes tens of milliseconds. The missing GPU time then shows up on the next line that waits for the device, such as `.item()` or `.cpu()`, making that line look slow instead.

Starting a timer without synchronizing charges the measured region for leftover work from earlier calls. Leaving the first call in the sample includes one-time startup costs like memory allocation, kernel selection, and lazy initialization, turning a 3 ms operation into 50 ms. Benchmarking bugs like these are the usual cause of speedups that disappear in production.
