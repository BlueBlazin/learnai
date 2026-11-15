There are two tempting designs. One appends every token to a per-request list
forever. The other gives each request an `asyncio.Queue(maxsize=...)` and does
`await queue.put(token)` in the engine loop. What does each of them do when one
client out of eight stops reading? What would the loop have to do instead of
appending, and instead of awaiting?

---

Make "is this sequence paused" a pure function of its buffer, and ask it when
planning a step:

```python
decode = [s.request.id for s in self._running if len(s._buffer) < self.max_buffered]
```

A paused sequence simply is not in the step, so appending after the step can never
overflow, and nothing ever awaits a client. Then handle the case this creates: with
no prefill and no decode there is no step to run, so wait on one `asyncio.Event`
that `submit`, `take`, `close`, and the stall callback all set.

---

The remaining pieces:

- After the step, for each stream that is still running and whose buffer is now
  full, arm `loop.call_at(loop.time() + stall_ms, self._end, stream, "stalled")` and
  keep the handle on the stream. Do this after handling `result.finished`, so a
  finished stream never gets a timer.
- `take()` swaps the buffer for an empty list, cancels the stream's stall timer, and
  sets the event.
- One method ends a stream early, for both reasons: return if the state is not
  `"open"`; set the state; drop the buffer; cancel the timer; remove the stream from
  the running list and `engine.release` it, or remove it from the waiting list; set
  the event. `close()` calls it with `"closed"`.
- After `await engine.step(...)`, look streams up in the running list as it is then:
  a stream closed or stalled during the step is in neither it nor `result.tokens`.
