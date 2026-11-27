`drain` is a synchronous method, so it cannot wait for requests to finish. It can only update state and schedule a timer. Think about which part of `Server` knows when nothing is pending and the engine is not inside a step, and at which exact point in its loop that holds.

---

Split the logic into three parts:

- `drain` sets `ready = False`, creates the drain future, schedules the deadline with `loop.call_at(loop.time() + grace_ms, self._terminate_pending)`, wakes the loop, and returns the future. `submit` checks `if not self.ready: raise Unavailable(...)` first.
- `_terminate_pending` runs at the deadline (possibly mid-step). It calls `engine.release` on every running sequence, clears both the running and waiting lists, and resolves each pending handle with `Terminated(tokens)`.
- In `run()`, the existing idle branch (`not self._waiting and not self._running`) is the only place where no work is pending and no `engine.step` is running. If a drain has started, resolve the drain future there, set `closed = True`, and `return`. Otherwise, block on `self._wake` as before.

---

Key details checked by the tests:

```python
if not self._waiting and not self._running:
    if self._drained is not None:
        if self._deadline is not None:
            self._deadline.cancel()
        self.closed = True
        self._drained.set_result(self._terminated)
        return
    self._wake.clear()
    await self._wake.wait()
    continue
```

- `drain` must call `self._wake.set()`, or an already-idle server will stay blocked in `run()`.
- After `await engine.step(...)`, look up handles in the current `self._running` list. Any sequence terminated mid-step is absent from both `self._running` and `result.tokens`.
- For `grace_ms == 0`, call `_terminate_pending()` directly inside `drain`. A timer scheduled for the current instant fires after a step completing at that instant.
- Resolve terminated handle futures with `set_result(Terminated(n))` and raise the exception inside `Handle.result()`. Using `set_exception` on a future that a caller never awaits triggers an asyncio warning.
