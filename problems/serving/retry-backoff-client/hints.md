Write the function as one loop whose body is: make an attempt, classify what came
back, decide whether to stop, compute the delay, sleep. Most of the work is the
classification. For each kind of outcome ask one question: does the client *know*
that the server did not do the work?

---

Keep three pieces of state: the start time, the number of retries made, and the
number of uniforms used (these two differ as soon as one delay comes from
`Retry-After`). After an attempt, tell the budget what happened, then check the stop
conditions in an order where each is cheap and independent: not retryable, no
attempts left, budget says no. Only then compute the delay, and compare the time
the retry *would start* with the deadline before sleeping, not after.

---

```python
delay = None if response is None else retry_after_s(response.headers)
if delay is None:
    delay = uniforms[drawn] * min(policy.cap_s, policy.base_s * 2**retries)
    drawn += 1
if (loop.time() - started) + delay >= policy.deadline_s:
    ...  # give up: return the response or raise the error
await asyncio.sleep(delay)
retries += 1
```

`retry_after_s` loops over the headers, compares `name.lower()` with
`"retry-after"`, and accepts the value only if `value.strip()` is all ASCII digits.
The cap goes inside the multiplication by the uniform, not around it.
