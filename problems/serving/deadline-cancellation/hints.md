`cancel()` and a deadline timer run while `run()` is suspended inside
`await engine.step(...)`. When the step returns, the loop's picture of who is running
may be out of date. Which data should the loop trust after the `await`: the lists it
built before, or the scheduler's state as it is now?

---

Write one method that ends a request, and send all three endings through it:
`_end(handle, outcome)` where `outcome` is a token count, a `Canceled`, or a
`DeadlineExceeded`. It returns `False` if the handle has already ended. Otherwise it
cancels the handle's deadline timer, removes the handle from wherever it is (the
waiting list, or the running list plus `engine.release(id)` when the outcome is an
exception), resolves the handle's future, and returns `True`. Then `cancel()` is one
line, the deadline is `loop.call_at(deadline, self._end, handle, DeadlineExceeded(...))`
set up in `submit`, and the loop calls `_end(handle, tokens)` for `result.finished`.

---

Three details finish it.

- In `submit`, check `deadline <= loop.time()` first and resolve the handle with
  `DeadlineExceeded` without queueing it.
- After the step, build the id-to-handle map from the running list *as it is then*;
  released sequences are gone from both it and `result.tokens`:

  ```python
  result = await self.engine.step(prefill=prefill, decode=decode)
  by_id = {h.request.id: h for h in self._running}
  for seq_id in result.tokens:
      by_id[seq_id]._tokens += 1
  for seq_id in result.finished:
      self._end(by_id[seq_id], by_id[seq_id]._tokens)
  ```

- Resolve the future with `set_result(outcome)` even for the exceptions, and let
  `result()` raise when the awaited value is an exception. A future that holds an
  exception nobody retrieves is reported by asyncio as an error.
