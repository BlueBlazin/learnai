Suppose three requests are waiting and two slots are free. Admitting the second request makes the step longer. Which requests pay that extra time, and whose deadline must you check before admitting it? Also, if a request cannot make its deadline by joining the current step, could it ever make its deadline in a later step?

---

Sort the waiting requests by deadline using Python's stable `sorted(waiting, key=...)`, mapping `None` to `math.inf`. Track the running total of prompt tokens admitted so far, and for each request compute:

```python
first_token = now_ms + step_ms(prefill_tokens + request.prompt_tokens, running)
```

Shed the request if `first_token` is strictly greater than its own deadline. Admit it if a slot is still free and `first_token` does not exceed the deadline of any already-admitted request. Otherwise, leave it in `waiting`. Do not `break` out of the loop, because later requests may still be admitted or shed.

---

Because the scan runs in earliest-deadline order, the tightest deadline among already-admitted requests is always `admit[0]`. The admission check therefore reduces to a single comparison:

```python
if first_token > deadline_of(request):
    shed.append(request)
elif len(admit) < free_slots and (not admit or first_token <= deadline_of(admit[0])):
    admit.append(request)
    prefill_tokens += request.prompt_tokens
```

Both comparisons treat equality as on time. After the loop, remove all requests in `admit` and `shed` from `waiting` in place, and return `(admit, shed)`.
