When `fn()` returns, how much of its work has the device actually finished? What must happen before you read the clock a second time so the interval includes the device work? And what might the device still be doing when you read the clock the *first* time?

---

Call `fn()` once and discard it as the warm-up. Then, for each of the `n` samples, wait for the device to finish, read the clock, call `fn()`, wait for the device to finish again, read the clock, and record the difference. The first wait ensures the device is idle at the start so the sample does not absorb the warm-up, the previous call, or any earlier queued work. The second wait ensures the interval includes the work queued by this call.

---

```python
device_api.sync()
start = device_api.clock()
fn()
device_api.sync()
samples.append(device_api.clock() - start)
```

For the return value, `statistics.median(samples)` sorts the durations, takes the middle value for an odd count, and averages the two middle values for an even count.
