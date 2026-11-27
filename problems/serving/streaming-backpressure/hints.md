Consider two common designs. The first appends every token to an unbounded per-request list. The second uses a bounded `asyncio.Queue(maxsize=...)` per request and calls `await queue.put(token)` inside the engine loop. What happens in each design when one client out of eight stops reading? How can the engine loop stop producing for that one client without blocking on it?

---

Make pausing a planning decision based on the buffer length:

```python
decode = [s.request.id for s in self._running if len(s._buffer) < self.max_buffered]
```

A paused sequence is omitted from the step, so appending after the step can never exceed `max_buffered`, and the loop never awaits a client. When both `prefill` and `decode` are empty, do not call `engine.step`. Instead, clear and wait on a single `asyncio.Event` that is set by `submit`, `take`, `close`, and the stall callback.

---

Wire up the remaining pieces:

- After handling `result.finished`, check every stream still in `self._running`. If its buffer is full and it has no stall timer yet, schedule `loop.call_at(loop.time() + self.stall_ms, self._end, stream, "stalled")` and store the handle on the stream. Doing this after `result.finished` ensures a finished stream never arms a stall timer.
- In `take()`, swap the buffer for a new empty list, cancel any active stall timer on that stream, and set the wake event.
- Use one helper method to end a stream early for both `"stalled"` and `"closed"`. Return immediately if `stream.state != "open"`. Otherwise set `stream.state`, clear the buffer, cancel any stall timer, remove the stream from `self._running` and call `engine.release(id)` (or remove it from `self._waiting`), and set the wake event.
- After `await engine.step(...)`, look up streams in `self._running` as it exists after the `await`. Any stream closed or stalled mid-step has already been removed from both `self._running` and `result.tokens`.
