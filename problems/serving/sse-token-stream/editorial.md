## The idea

Server-sent events are the simplest streaming protocol on the web: an ordinary HTTP
response that never has to end, whose body is lines of text. The format (from the
WHATWG HTML standard) has four rules that matter here:

- the content type is `text/event-stream` and the encoding is always UTF-8;
- a line `data: something` adds `something` to the current event (one space after the
  colon is optional and is stripped);
- a **blank line dispatches the event**; without it the client keeps accumulating, and
  an event that is still incomplete when the stream ends is discarded;
- several `data:` lines in one event are joined with newlines, which is why a payload
  must never contain a raw line break. `json.dumps` escapes them, so one JSON document
  per event is always safe.

`[DONE]` is not part of the standard. It is the convention OpenAI's API established for
"the stream ended on purpose", which a client cannot otherwise tell apart from a
dropped connection.

On the server side, ASGI maps this directly onto messages: one `http.response.start`,
then any number of `http.response.body` messages with `more_body: True`. The server
writes each one to the socket when it is sent (with chunked transfer encoding, since
there is no content length), so "send one message per token" *is* the flush.

That leaves two decisions with consequences:

**Validate before the first byte.** The status line goes out with
`http.response.start`. Everything that can produce a 4xx (path, method, body) must be
decided before that, and only then is the generator created.

**Listen for the disconnect concurrently.** A handler blocked in `async for` on a slow
model cannot notice anything. The disconnect arrives as a message on `receive()`, so
the stream and a watcher run as two tasks, and the first to finish cancels the other.
Canceling the pump throws `CancelledError` into whatever it is awaiting, including the
generator's own `await`, which unwinds the generator; `aclose()` afterwards covers the
case where the generator was suspended at a `yield`. In a real server, that
cancellation is what frees the request's slot in the running batch.

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

`read_body`, `send_json`, and the `BadRequest` exception are those of the starter. `json.loads` accepts
bytes and raises a `ValueError` subclass both for bad JSON and for bytes that are not
UTF-8, so one `except` covers both.

## Common mistakes

- **Buffering.** Collecting the events and sending them in one body message (or
  returning a list to a framework's non-streaming response) gives a correct body that
  arrives all at once. The same symptom appears in production with every byte of the
  app correct, when a proxy or a compression middleware buffers the response; that is
  why streaming servers also send `cache-control: no-cache` and often
  `x-accel-buffering: no`.
- **Wrong content type.** A browser's `EventSource` refuses a response that is not
  `text/event-stream`, and proxies use the type to decide whether to buffer.
- **Missing blank line.** `data: ...\n` with a single newline never dispatches; the
  client sees one giant event at the end, with all payloads joined by newlines.
- **No terminal event.** Without `[DONE]` the client cannot distinguish a finished
  generation from a connection that dropped after the fourth token.
- **Ignoring the disconnect.** The stream "works" in every demo. Under load, each
  closed browser tab keeps generating to `max_tokens`.
- **Trusting `send` to fail.** Depending on the server and version, sending to a
  closed connection raises, or does nothing. Only `http.disconnect` is reliable.
- **200 on bad input.** Coercing a missing prompt to `""` or `"max_tokens": "3"` to
  `3` hides client bugs. A cousin of this mistake: `isinstance(True, int)` is true, so
  `"max_tokens": true` passes a naive type check and becomes 1.
- **500 on bad JSON.** Forgetting to catch the decoding error turns a client mistake
  into a server error, with a stack trace in the logs for every malformed request.
- **Hand-built JSON.** `f'{{"token": "{token}"}}'` breaks on the first token with a
  quote or a newline, and newline tokens are common.
- **404 for a wrong method.** Matching on the pair (method, path) loses the
  distinction between "no such resource" and "wrong verb"; a 405 must also say which
  methods are allowed.

## Complexity

Constant work and memory per token; nothing is accumulated. The handler holds two
tasks per open stream, so the cost of an idle connection is a few kilobytes.

## In the wild

- **vLLM**'s OpenAI-compatible server and **llama.cpp**'s server stream exactly this
  shape: `data: {json}` chunks and a final `data: [DONE]`.
- **TGI** (text-generation-inference) streams SSE events with one token per event and
  marks the end inside the last event (it carries the full generated text) instead of
  using `[DONE]`.
- **Starlette**'s `StreamingResponse`, which FastAPI servers such as vLLM's use, is the
  same structure as the reference: it runs the body iterator and a
  `receive()`-listener side by side and cancels the stream on `http.disconnect`.
- Two things real servers add: a **comment line** (a line starting with `:`) every
  15 seconds or so as a keep-alive, so that idle proxies do not cut a slow stream, and
  **backpressure**: `await send(...)` waits while the client's socket buffer is full, so
  a slow reader slows the pump instead of growing a queue. One more subtlety when the
  tokens come from a byte-level tokenizer: a token can end in the middle of a
  multi-byte character, so servers hold back incomplete UTF-8 sequences until the next
  token completes them.
