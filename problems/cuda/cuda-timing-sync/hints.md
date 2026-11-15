When `fn()` returns, how much of its work has the device done? What has to happen
before reading the clock a second time, so that the difference of the two readings
contains the work? And what is the device still doing at the moment you read the
clock the *first* time?

---

Call `fn()` once and throw the call away: that is the warm-up. Then, for each of the
`n` samples: wait for the device, read the clock, call `fn()`, wait for the device,
read the clock again, and keep the difference. The first wait makes sure the timer
starts with an idle device, so that the sample does not pay for the warm-up, for the
previous call, or for work queued before `time_region` was called. The second wait
puts this call's work inside the interval.

---

```python
device_api.sync()
start = device_api.clock()
fn()
device_api.sync()
samples.append(device_api.clock() - start)
```

For the result, `statistics.median(samples)` does what the statement defines: it
sorts, takes the middle value for an odd count, and averages the two middle values
for an even count.
