Suppose three requests are waiting and two slots are free. Admitting the second one
makes the step longer. Who pays for that, and whose deadline do you therefore have
to look at before you admit it? And if a request cannot make its deadline by joining
the step you are planning right now, could it make it by joining a later one?

---

Sort the waiting requests by deadline with a stable sort (`sorted(waiting, key=...)`
with `math.inf` for `None`), keep a running total of the prompt tokens admitted so
far, and for each request compute

```python
first_token = now_ms + step_ms(prefill_tokens + request.prompt_tokens, running)
```

Shed it if `first_token` is after its own deadline. Admit it if a slot is left and
`first_token` is not after the deadline of anything already admitted. Otherwise
leave it. Do not `break` anywhere: requests behind a waiting one may still be
admitted or shed.

---

Because the scan is in deadline order, the tightest deadline among the admitted
requests is that of the first one, so the second test is a single comparison:

```python
if first_token > deadline_of(request):
    shed.append(request)
elif len(admit) < free_slots and (not admit or first_token <= deadline_of(admit[0])):
    admit.append(request)
    prefill_tokens += request.prompt_tokens
```

Both comparisons keep equality on the on-time side. Afterwards remove `admit` and
`shed` from `waiting` (the list must be modified in place) and return the pair.
