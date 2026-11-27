## The idea

Server-Sent Events (SSE) is a lightweight streaming protocol built on a standard HTTP response whose body consists of text lines. Four rules from the WHATWG HTML specification matter here:

- The content type is `text/event-stream`, and the encoding is always UTF-8.
- A line `data: something` appends `something` to the current event's payload (one optional space after the colon is stripped).
- A **blank line dispatches the event**. Without it, the client keeps buffering, and any incomplete event left when the stream closes is discarded.
- Multiple `data:` lines in one event are joined with newlines. Payloads must therefore never contain raw line breaks. Using `json.dumps` escapes newlines automatically, so one JSON object per event is always safe.

`[DONE]` is not part of the SSE standard. It is the convention popularized by the OpenAI API to signal that generation finished normally, letting the client distinguish a clean completion from a dropped connection.

ASGI maps SSE streaming directly onto messages. Send one `http.response.start`, followed by `http.response.body` messages with `more_body: True`. Because there is no `Content-Length` header, the ASGI server uses chunked transfer encoding and writes each body message to the socket immediately. Sending one message per token is what flushes the stream.

Two architectural choices make the handler reliable:

**Validate before the first byte.** The HTTP status line is committed when you send `http.response.start`. All checks that can produce a 4xx response (path, method, and request body) must finish before that call, and `token_source` should only be invoked once the request is valid.

**Watch for disconnects concurrently.** A coroutine blocked in `async for` waiting on a slow model cannot notice a closed connection on its own. Because disconnects arrive as messages from `receive()`, run the token pump and a disconnect watcher as two concurrent tasks and cancel the loser as soon as either finishes. Canceling the pump injects `CancelledError` into whatever it is awaiting, unwinding the generator. Calling `await tokens.aclose()` in a `finally` block also cleans up the generator if it was paused at a `yield`. In a production server, that cleanup releases the request's slot in the running batch.

## Reference solution (Python)

```python
def make_app(token_source):
    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        if scope["path"] != "/v1/generate":
            await send_json(send, 404, {"error": "not found"})
            return
        if scope["method"] != "POST":
            await send_json(send, 405, {"error": "method not allowed"}, [(b"allow", b"POST")])
            return
        body = await read_body(receive)
        if body is None:
            return
        try:
            prompt, max_tokens = parse_request(body)
        except BadRequest as exc:
            await send_json(send, 400, {"error": str(exc)})
            return
        await stream_tokens(token_source(prompt, max_tokens), receive, send)

    return app


def parse_request(body):
    try:
        payload = json.loads(body)
    except ValueError:
        raise BadRequest("the body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise BadRequest("the body must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise BadRequest("'prompt' is required and must be a string")
    max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("'max_tokens' must be an integer >= 1")
    return prompt, max_tokens


async def stream_tokens(tokens, receive, send):
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")],
    })

    async def pump():
        index = 0
        async for token in tokens:
            data = json.dumps({"token": token, "index": index})
            await send(sse_message(data, more_body=True))
            index += 1
        await send(sse_message("[DONE]", more_body=False))

    async def disconnected():
        while (await receive())["type"] != "http.disconnect":
            pass

    tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(disconnected())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await tokens.aclose()


def sse_message(data, *, more_body):
    return {
        "type": "http.response.body",
        "body": f"data: {data}\n\n".encode(),
        "more_body": more_body,
    }
```

`DEFAULT_MAX_TOKENS`, `read_body`, `send_json`, and `BadRequest` are provided in the starter file. `json.loads` accepts `bytes` directly and raises a `ValueError` subclass for both invalid JSON and invalid UTF-8, so a single `except ValueError` handles both cases.

## Common mistakes

- **Buffering events.** Collecting events in a list and sending them in one body message produces the right bytes but delivers them all at once after generation ends. The same symptom happens in production when a reverse proxy or compression middleware buffers the stream, which is why servers also send `cache-control: no-cache` and often `x-accel-buffering: no`.
- **Wrong content type.** Browsers' `EventSource` rejects responses that do not use `text/event-stream`, and proxies rely on that header to disable buffering.
- **Missing blank line.** Sending `data: ...\n` with only one newline prevents the client from dispatching each event. Instead, the client accumulates every payload into one giant event separated by newlines.
- **Missing terminal event.** Without `data: [DONE]\n\n`, the client cannot tell whether generation completed normally or the connection dropped mid-stream.
- **Ignoring client disconnects.** The endpoint appears to work in simple demos, but under real traffic every closed browser tab keeps generating until `max_tokens`.
- **Relying on `send()` to raise.** Depending on the ASGI server and version, writing to a closed connection may raise an error or silently do nothing. Listening for `http.disconnect` on `receive()` is the only reliable signal.
- **Returning 200 on invalid input.** Coercing a missing prompt to `""` or `"max_tokens": "3"` to `3` masks client bugs. In Python, `isinstance(True, int)` is `True`, so `"max_tokens": true` passes a naive `isinstance(x, int)` check and silently becomes `1`.
- **Returning 500 on malformed JSON.** Letting `json.loads` raise unhandled turns a client mistake into an internal server error and floods server logs with tracebacks.
- **Building JSON with string formatting.** Formatting `f'{{"token": "{token}"}}'` breaks on the first token containing a quote or newline, both of which occur frequently in LLM output.
- **Returning 404 for a wrong HTTP method.** Matching on `(method, path)` together collapses "resource does not exist" and "method not allowed" into one status. A `405` response must also include the `allow` header.

## Complexity

The handler uses $O(1)$ work and $O(1)$ memory per token because nothing is buffered. Each open stream holds two lightweight `asyncio` tasks, costing only a few kilobytes per idle connection.

## In the wild

- **vLLM**'s OpenAI-compatible server and **llama.cpp**'s server stream this exact wire format: `data: {json}` chunks followed by `data: [DONE]`.
- **TGI** (Text Generation Inference) also streams one token per SSE event, but signals completion inside the final JSON event (carrying the full generated text) instead of sending `[DONE]`.
- **Starlette**'s `StreamingResponse`, used under the hood by FastAPI servers including vLLM, follows the same two-task architecture. It runs the response iterator alongside a `receive()` listener and cancels the stream on `http.disconnect`.
- Production servers add two more mechanisms. First, they send an SSE **comment line** (starting with `:`) every 15 seconds or so as a keep-alive so idle proxies do not close slow streams. Second, they rely on **backpressure**. When the client's socket buffer fills up, `await send(...)` pauses, slowing the producer rather than growing an unbounded queue. When using byte-level tokenizers, servers also buffer incomplete multi-byte UTF-8 sequences until the next token completes the character.
