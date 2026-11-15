Two questions decide everything. What do you know about a call *before* you make it,
and what do you only know after? The price is known before; the confidence only
after. Which of the two checks therefore belongs in front of `model.complete`?

---

One loop over `enumerate(models)` with a running `spent`. At the top: if
`spent + model.cost > budget`, stop without calling. Then call with
`[{"role": "user", "content": prompt}]`, add the cost, and compute the confidence as
`sum(logprobs) / len(logprobs)`. Remember the content, name, and confidence of this
reply before you decide anything: if the budget stops the next stage, this is the
answer you return.

---

The decision after a call:

```python
last = i == len(models) - 1
if last or confidence >= thresholds[i]:
    accepted = True
    break
```

`thresholds` has one entry fewer than `models`, so test `last` first. `accepted`
stays `False` in exactly one situation: the loop ended on the budget check.
