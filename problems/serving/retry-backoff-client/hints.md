Structure the function as a single loop: make an attempt, classify the outcome, decide whether to stop, compute the delay, and sleep. Most of the subtlety is in the classification. For each outcome, ask whether the client *knows* that the server did not execute the request.

---

Track three pieces of state: the call start time, the number of retries made, and the number of uniforms consumed (the last two diverge as soon as a delay comes from `Retry-After`). After each attempt, update the budget first, then check the cheap stop conditions: not retryable, no attempts remaining, or budget disallows retry. Only if all three pass should you compute the delay and check whether the next retry would start before the deadline.

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

In `retry_after_s`, iterate over the headers, compare `name.lower()` against `"retry-after"`, and accept the value only if `value.strip()` contains solely ASCII digits. Apply `min(cap_s, ...)` before multiplying by `uniforms[drawn]`, not after.
