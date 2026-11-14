Both response modes report the same generation: the same text, the same finish
reason, the same usage. What are the three numbers you need at the end of a
generation, and which of them do you know before the first piece arrives? Compute
them in one place and let the two modes differ only in packaging.

---

Validate everything before the first byte of the response, with `payload.get(...)`
so that a missing field becomes a `BadRequest` with its `param` and not a `KeyError`
(remember that `isinstance(True, int)` is true). Then build what every object of the
response shares, once per request: `{"id": "chatcmpl-" + uuid.uuid4().hex, "created":
int(time.time()), "model": request.model}`. The finish reason is a comparison of the
piece count with the limit: `"length" if count == max_tokens else "stop"`, which is
`"stop"` whenever the limit is `None`. Not streamed: collect the pieces, then send one
object. Streamed: send the `200` with `text/event-stream`, then one event per chunk.

---

The stream is five kinds of events, in order, each sent as its own body message
`data: <json>\n\n`:

```python
await send(chunk(choice({"role": "assistant", "content": ""})))
count = 0
async for piece in pieces:
    await send(chunk(choice({"content": piece})))
    count += 1
await send(chunk(choice({}, finish_reason(count, request.max_tokens))))
if request.include_usage:
    await send(chunk([], usage(prompt_tokens, count)))     # choices is empty here
await send(sse_message("[DONE]", more_body=False))
```

where `choice(delta, reason=None)` builds `[{"index": 0, "delta": delta,
"finish_reason": reason}]` and `chunk` adds the shared fields, the object type, and,
when usage was requested, `"usage": None` on every chunk but the last. Run this pump
next to a task that waits for `http.disconnect` on `receive()`, exactly as in
`sse-token-stream`, and call `await pieces.aclose()` in the `finally`.
