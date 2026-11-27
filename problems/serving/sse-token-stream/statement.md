Build a raw ASGI HTTP application that streams a model's tokens to the client as Server-Sent Events (SSE), without using a web framework. The endpoint must reject invalid requests before the stream starts, flush every token as soon as it is generated, and stop generation promptly when the client disconnects.

![This problem is the Token stream, here server-sent events flushed one token at a time.](assets/map.svg)

## Specification

```python
def make_app(token_source):
    async def app(scope, receive, send): ...
    return app
```

`make_app` returns an [ASGI](https://asgi.readthedocs.io/en/latest/specs/www.html) HTTP application. The tests call `make_app` with their own `token_source`, serve the returned app with uvicorn on a local socket, and send requests over real HTTP.

**The token source.** Calling `token_source(prompt, max_tokens)` returns an async generator that yields generated tokens as `str`, one at a time. It may pause for any length of time between tokens, enforces `max_tokens` itself, and does not raise exceptions. A token can be any string, including the empty string or strings containing quotes, newlines, or non-ASCII characters. Call `token_source` once per valid request and never for an invalid one.

**Routing.** Check each HTTP request in this order:

1. Any path other than `/v1/generate` returns status `404`, regardless of method. The query string is not part of the path (`scope["path"]` excludes it).
2. Any method other than `POST` on `/v1/generate` returns status `405` with the response header `allow: POST`.
3. A `POST` request with an invalid body returns status `400`.
4. Otherwise, return status `200` and stream the tokens.

Every `404`, `405`, and `400` response must have content type `application/json` and a JSON object body `{"error": "<message>"}` with any non-empty message string.

**The request body.** The body may arrive in several `http.request` messages and is complete when `more_body` is `False` or missing. A body is valid if it is UTF-8 encoded JSON representing an object with these fields:

- `"prompt"`: required string (the empty string is allowed).
- `"max_tokens"`: optional integer `>= 1`, defaulting to `16`. Strings, `null`, numbers with a fractional part such as `2.5`, and the booleans `true` and `false` are invalid.

Other keys are ignored. Empty bodies, invalid UTF-8, invalid JSON, and JSON values that are not objects are all invalid.

**The stream.** A valid request returns status `200` with content type `text/event-stream` (an optional `charset=utf-8` parameter is allowed). The response body is UTF-8 encoded in the [Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html) format:

- One event per token in order, whose data is the JSON object `{"token": <the token>, "index": <i>}` with `i` counting from `0`.
- One final event whose data is the literal text `[DONE]`.
- The end of the response (a body message with `more_body` set to `False`).

Each event is a single `data: <payload>` line followed by a blank line (`data: <payload>\n\n`). The JSON payload must stay on one line, as `json.dumps` produces, with any key order or spacing. Do not include `event:`, `id:`, or `retry:` fields.

**Flushing.** Send each token's event in its own `http.response.body` message with `more_body` set to `True` as soon as the source yields that token, before requesting the next token. The ASGI server writes each body message to the socket immediately, which makes the stream incremental.

**Client disconnect.** If the client closes the connection before the stream finishes, stop promptly. Request no further tokens, close the generator with `await tokens.aclose()`, and return. Detect the disconnect through `receive()`. After the request body has been read, the next `await receive()` blocks until the client disconnects and then returns `{"type": "http.disconnect"}`. Do not rely on `send()` failing, because ASGI only recommends that servers raise `OSError` on a closed connection, and the server used here silently discards writes instead.

**Other scope types.** If `scope["type"]` is not `"http"` (for example, the `"lifespan"` scope sent at startup), return without sending anything.

Use only the Python standard library (`asyncio`, `json`). Do not use Starlette or any other web framework.

## Example

With a token source that yields `"Hello"`, `","`, `" world"`, `"!"`:

```
POST /v1/generate
{"prompt": "Say hello", "max_tokens": 4}
```

The app calls `token_source("Say hello", 4)` and sends this response:

```
HTTP/1.1 200 OK
content-type: text/event-stream

data: {"token": "Hello", "index": 0}

data: {"token": ",", "index": 1}

data: {"token": " world", "index": 2}

data: {"token": "!", "index": 3}

data: [DONE]

```

This response is sent as five body messages, one per event. The blank line after each `data:` line terminates the event so the client's SSE parser dispatches it immediately.

Example error responses (your error messages may differ):

```
POST /v1/generate   {"prompt": 5}    ->  400  {"error": "'prompt' is required and must be a string"}
GET  /v1/generate                    ->  405  {"error": "method not allowed"}   allow: POST
POST /v2                             ->  404  {"error": "not found"}
```

## Constraints

- Request bodies are at most 64 KiB, and a generation produces at most 1000 tokens.
- Events are parsed according to the SSE standard and compared exactly after JSON decoding, checking the tokens, their order, their indices, and `[DONE]` as the final event.
- Incremental delivery is tested by ordering rather than wall-clock timing. The test's token source does not yield token `i + 1` until the client has received the event for token `i`. A server that buffers events stalls and fails after a 2-second timeout.
- For the disconnect test, the source yields a token every 5 ms for 200 tokens. The client reads three events and closes the connection. The test fails if the source is drained to the end, or if it is neither closed nor exhausted within 5 seconds.
- The app must handle multiple sequential and concurrent requests without blocking the event loop (no `time.sleep` or synchronous I/O).

## Why this matters

Users judge an LLM service by time to first token and stream smoothness, both of which depend on this handler. If any layer buffers the response (the app, middleware, or a proxy that does not recognize the content type), the output arrives all at once after generation finishes and the service feels broken even though every byte is right.

Disconnect handling directly controls compute waste. If the server does not detect that a client hung up and cancel the generator, the model keeps holding a batch slot and burning GPU cycles for an abandoned request. Finally, because the HTTP status line is sent before the first token, all request validation must happen up front. Once `200 OK` goes out, HTTP status codes can no longer report client errors.
