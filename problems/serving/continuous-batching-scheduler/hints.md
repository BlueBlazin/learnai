Track two collections across steps: the requests waiting for a slot and the sequences currently holding one. After `engine.step` returns, think about which of those collections change and what tells you how to update them. Remember that a sequence admitted in the current step was prefilled, not decoded.

---

Keep a `deque` of waiting requests and a list `running` of live sequence ids in admission order. In each iteration of `while True`:

1. If `waiting`, `running`, and `queue` are all empty, block on `await queue.get()` and append the request to `waiting`.
2. Drain `queue` with `get_nowait()` while `not queue.empty()`.
3. While `waiting` is non-empty and `len(running) + len(prefill) < max_running`, pop from the left of `waiting`, call `engine.admit(request)`, and append `(request.id, request.prompt_tokens)` to `prefill`.
4. Call `await engine.step(prefill=prefill, decode=running)` using `running` as it stood *before* this step's admissions.

Steps 1 and 2 guarantee that step 4 never runs an empty batch.

---

Updating `running` right after the step avoids the most common bugs. Newly prefilled sequences join `running` only after the step returns (so they decode from the next step onward), and every id in `result.finished` is removed immediately (so its slot is free before the next admission check):

```python
result = await engine.step(prefill=prefill, decode=running)
finished = set(result.finished)
running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
```

If a request has only one output token, it appears in both `prefill` and `result.finished` on the same step, and this comprehension handles it without a special case.
