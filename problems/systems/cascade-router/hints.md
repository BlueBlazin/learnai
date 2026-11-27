Separate what you know *before* calling a model from what you learn *after*. You know a model's cost before the call, and you learn its confidence only from the reply. Which check belongs before `model.complete`?

---

Loop over `enumerate(models)` while tracking `spent`. At the top of each iteration, stop if `spent + model.cost > budget`. Otherwise, call the model with `[{"role": "user", "content": prompt}]`, add `model.cost` to `spent`, and compute `confidence = sum(logprobs) / len(logprobs)`. Record the reply's `content`, `model.name`, and `confidence` right away so they are ready to return if the budget stops the next iteration.

---

After recording the reply, decide whether to accept and stop:

```python
last = i == len(models) - 1
if last or confidence >= thresholds[i]:
    accepted = True
    break
```

Because `thresholds` has one fewer entry than `models`, check `last` before indexing `thresholds[i]`. The `accepted` flag stays `False` only when the loop stops at the budget check.
