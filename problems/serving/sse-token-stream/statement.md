Build the endpoint that streams a model's tokens to the client as server-sent events,
as a raw ASGI application with no web framework. Producing the right bytes is the easy
part; the work is in flushing every token the moment it exists, rejecting bad requests
before the stream starts, and stopping generation when the client hangs up.

## Specification

```python
def make_app(token_source):
    async def app(scope, receive, send): ...
    return app
```

`make_app` returns an [ASGI](https://asgi.readthedocs.io/en/latest/specs/www.html)
HTTP application. The tests call `make_app` with their own `token_source`, serve the
returned app with uvicorn on a local socket, and talk to it over real HTTP.

**The token source.** `token_source(prompt, max_tokens)` returns an async generator
that yields the generated tokens as `str`, one at a time. It may pause for any length
of time before a token, it honors `max_tokens` itself, and it does not raise. A token
can be any string, including the empty string and strings with quotes, newlines, or
non-ASCII characters. Call `token_source` once per valid request and never for an
invalid one.

**Routing**, checked in this order:

1. A path other than exactly `/v1/generate` gets status `404`, whatever the method.
   The query string is not part of the path (`scope["path"]` excludes it).
2. A method other than `POST` on `/v1/generate` gets status `405` and the response
   header `allow: POST`.
3. A `POST` whose body is invalid gets status `400`.
4. Otherwise the response is the token stream, with status `200`.

Every `404`, `405`, and `400` response has the content type `application/json` and a
body that is a JSON object `{"error": "<message>"}` with a non-empty message of your
choice.

**The request body** is valid if it is a JSON object (UTF-8) with

- `"prompt"`: required, a string (the empty string is allowed);
- `"max_tokens"`: optional, default `16`; if present, a JSON integer `>= 1`. Strings,
  `null`, numbers with a fraction such as `2.5`, and the booleans `true` and `false`
  are invalid.

Other keys are ignored. Anything else is invalid: a body that is not JSON or not UTF-8,
an empty body, or JSON that is not an object. The request body may arrive in several
`http.request` messages; it is complete when `more_body` is false or missing.

**The stream** is a `200` response with the content type `text/event-stream` (a
`charset=utf-8` parameter is allowed) whose body is, in the
[server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
format, encoded as UTF-8:

- one event per token, in order, whose data is the JSON object
  `{"token": <the token>, "index": <i>}` with `i` counting from 0;
- then one final event whose data is the literal text `[DONE]`;
- then the end of the response (a body message with `more_body` false).

An event is one line `data: <payload>` followed by a blank line, that is,
`data: <payload>\n\n`. The JSON must be on a single line (as `json.dumps` produces);
events carry no `event:`, `id:`, or `retry:` fields. The key order and spacing of the
JSON are free.

**Flushing.** The event for a token is sent in its own `http.response.body` message
(with `more_body` true) as soon as the source yields that token, before the next token
is requested from the source. The server writes each body message to the socket
immediately, so this is what makes the stream incremental.

**Client disconnect.** If the client closes the connection before the stream is
finished, the app stops promptly: it requests no further tokens and closes the
generator (`await tokens.aclose()`), then returns. The disconnect is announced by
`receive()`: once the request body has been read, the next `await receive()` blocks
until the client is gone and then returns `{"type": "http.disconnect"}`. Do not rely
on `send` failing: the ASGI specification only says a server *should* raise `OSError`
when sending on a closed connection, and the server used here silently discards the
message instead.

**Other scope types.** For a `scope["type"]` other than `"http"` (the server sends a
`"lifespan"` scope at startup), return without sending anything.

Use only the standard library (`asyncio`, `json`); do not use Starlette or another
framework.

## Example

With a token source that yields `"Hello"`, `","`, `" world"`, `"!"`:

```
POST /v1/generate
{"prompt": "Say hello", "max_tokens": 4}
```

the source is called as `token_source("Say hello", 4)`, and the response is

```
HTTP/1.1 200 OK
content-type: text/event-stream

data: {"token": "Hello", "index": 0}

data: {"token": ",", "index": 1}

data: {"token": " world", "index": 2}

data: {"token": "!", "index": 3}

data: [DONE]

```

sent as five body messages, one per event. The blank line after each `data:` line is
what ends the event; a client's parser hands an event to the application only when it
sees that blank line.

Errors, with the reference's messages (yours may differ):

```
POST /v1/generate   {"prompt": 5}    ->  400  {"error": "'prompt' is required and must be a string"}
GET  /v1/generate                    ->  405  {"error": "method not allowed"}   allow: POST
POST /v2                             ->  404  {"error": "not found"}
```

## Constraints

- Request bodies are at most 64 KiB; a generation has at most 1000 tokens.
- Events are parsed with the standard's rules and compared exactly after JSON
  decoding: the tokens, their order, the indices, and `[DONE]` as the last event.
- Incremental delivery is checked by order, not by timing: the test's source does not
  generate token `i + 1` until the test has received the event for token `i`. A server
  that holds events back therefore stalls, and fails after a 2-second bound.
- For the disconnect check, the source yields a token every 5 ms for 200 tokens; the
  client reads three events and closes the connection. The case fails if the source
  is drained to its end, or is neither closed nor exhausted within 5 seconds.
- The app must handle several requests, one after another and concurrently, and must
  not block the event loop (no `time.sleep`, no synchronous I/O).

## Why this matters

Users judge an LLM service by the time to the first token and by the smoothness of
the stream, and both are decided in this handler. A response that is buffered anywhere
(in the app, in a middleware, in a proxy that does not recognize the content type)
arrives all at once after the generation is over, and the product feels broken even
though every byte is correct. The disconnect path is where the money is: a model
keeps occupying a batch slot and burning GPU time for every abandoned request unless
the server notices that the reader is gone and cancels the generation. And because
the status line is sent before the first token, validation has to happen up front:
after the `200`, the only way left to report an error is inside the stream.
