`drain` cannot do its job by waiting: it is a plain method and must return at once.
What it can do is change state and arm a timer. Which piece of code is in a position
to notice that nothing is pending and the engine is idle, and at exactly which point
of its loop is that true?

---

Three pieces.

- `drain` sets `ready = False`, creates the future, arms the deadline with
  `loop.call_at(loop.time() + grace_ms, self._terminate_pending)`, wakes the loop, and
  returns the future. `submit` starts with `if not self.ready: raise Unavailable(...)`.
- `_terminate_pending` runs at the deadline, possibly in the middle of a step:
  `engine.release` every running sequence, empty the running and waiting lists, and
  resolve each handle with `Terminated(tokens)`.
- The loop already has a branch for "nothing waiting and nothing running". That is
  the one place where the server is idle: the loop is the only caller of
  `engine.step` and it is not inside one. If a drain has started, resolve the future
  there and `return`; otherwise block as before.

---

The details that the tests look at:

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

- `drain` must `self._wake.set()`, or an idle server never notices the drain.
- After `await engine.step(...)`, look handles up in the running list as it is then:
  a sequence terminated during the step is in neither the list nor `result.tokens`.
- For `grace_ms == 0` call `_terminate_pending()` directly inside `drain`. A timer
  for the current instant fires after a step that completes at this instant.
- Resolve handle futures with `set_result(Terminated(n))` and raise in `result()`; a
  future holding an exception that nobody retrieves is reported by asyncio.
