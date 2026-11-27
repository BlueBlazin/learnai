Both streaming and non-streaming responses describe the same generation with the same text, finish reason, and token usage. Which of those numbers do you know before the first piece arrives, and which do you know only after the stream ends? Compute the shared state once and let the two modes differ only in how they format the output.

---

Validate the entire request before sending any response bytes, using `payload.get(...)` so missing fields become a `BadRequest` with the right `param` instead of a `KeyError` (and remember that `isinstance(True, int)` is `True` in Python). Build the shared response fields once per request: `{"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": request.model}`. Compute `finish_reason` as `"length" if count == max_tokens else "stop"`, which naturally yields `"stop"` when `max_tokens` is `None`. For non-streaming requests, collect all pieces and send one JSON object. For streaming requests, send the `200` headers with `text/event-stream` and emit one SSE event per chunk.

---

Emit the stream as five event types in order, each sent in its own `http.response.body` message formatted as `data: <json>\n\n`:

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

Here `choice(delta, reason=None)` returns `[{"index": 0, "delta": delta, "finish_reason": reason}]`, and `chunk` attaches the shared fields, `"object": "chat.completion.chunk"`, and `"usage": None` on every non-usage chunk when `include_usage` is enabled. Run this pump alongside a task waiting for `http.disconnect` on `receive()`, just as in `sse-token-stream`, and call `await pieces.aclose()` in a `finally` block.
