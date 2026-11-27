`cancel()` and deadline timers can fire while `run()` is suspended inside `await engine.step(...)`. When the step returns, the set of running requests may have shrunk. Think about whether the loop should trust the lists it built before the step or the scheduler's live state after the `await`.

---

Route all three request endings through a single `_end(handle, outcome)` method, where `outcome` is either a token count, a `Canceled` instance, or a `DeadlineExceeded` instance. If the handle has already ended, return `False`. Otherwise, cancel its deadline timer, remove it from `_waiting` or `_running` (calling `engine.release(id)` if it was running and `outcome` is an exception), resolve its future, and return `True`. Then `cancel()` is a one-line call to `_end`, deadlines use `loop.call_at(deadline, self._end, handle, DeadlineExceeded(...))` in `submit`, and `run()` calls `_end(handle, tokens)` for each id in `result.finished`.

---

- In `submit`, check `deadline <= loop.time()` first and immediately resolve the handle with `DeadlineExceeded` without adding it to `_waiting`.
- After `await self.engine.step(...)`, build `by_id` from `self._running` as it stands now, since released sequences are absent from both `self._running` and `result.tokens`:

  ```python
  result = await self.engine.step(prefill=prefill, decode=decode)
  by_id = {h.request.id: h for h in self._running}
  for seq_id in result.tokens:
      by_id[seq_id]._tokens += 1
  for seq_id in result.finished:
      self._end(by_id[seq_id], by_id[seq_id]._tokens)
  ```

- Resolve the future with `set_result(outcome)` even when `outcome` is an exception, and raise it inside `result()`. Using `set_exception` on a future that no caller ever awaits triggers an asyncio "Future exception was never retrieved" warning.
