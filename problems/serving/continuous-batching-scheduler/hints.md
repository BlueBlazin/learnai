The loop needs to remember two things between steps: who is waiting for a slot, and
who holds one. After a step returns, which of those change, and what tells you how?
(Look at what `engine.step` returns, and remember that a sequence admitted in this
step was prefilled, not decoded.)

---

Keep a `deque` of waiting requests and a list `running` of ids in admission order. One
iteration of `while True`:

1. if both are empty and the queue is empty, `await queue.get()` and append the result;
2. drain the queue with `get_nowait()` while `not queue.empty()`;
3. pop from the left of `waiting` while `len(running) + len(prefill) < max_running`,
   calling `engine.admit(request)` and appending `(request.id, request.prompt_tokens)`
   to this step's `prefill`;
4. `await engine.step(prefill=prefill, decode=running)`, passing `running` as it was
   *before* this step's admissions.

Steps 1 and 2 guarantee that the step in 4 is never empty.

---

The update after the step is where the classic mistakes live. The new sequences join
`running` only now (so they are decoded from the next step on), and everything in
`result.finished` leaves in the same statement (so its slot is free when the next
iteration counts):

```python
result = await engine.step(prefill=prefill, decode=running)
finished = set(result.finished)
running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
```

A request with one output token finishes in its prefill step: it is in `prefill` and
in `result.finished` at once, and this line handles it without a special case.
