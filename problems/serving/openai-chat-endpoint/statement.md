Implement the endpoint that almost every LLM client speaks: `POST
/v1/chat/completions`, in the shape of OpenAI's Chat Completions API, both as a single
JSON response and as a stream of chunks. Generating text is the model's job; the
endpoint's job is the protocol, and clients break on its details: the finish reason,
the token usage, where the role goes in a stream, the usage chunk, and `[DONE]`.

## Specification

```python
def make_app(lm):
    async def app(scope, receive, send): ...
    return app
```

`make_app` returns an ASGI application; the tests serve it with uvicorn on a local
socket and talk to it over HTTP. The starter is written for the raw ASGI interface of
[`sse-token-stream`](../sse-token-stream/statement.md) and provides `read_body`,
`send_json`, `render_prompt`, and `error_body`. Starlette may be used instead.

**The model.** `lm` is a [`ScriptedLM`](../../../python/src/whetstone/kit/scriptedlm.py):

- `lm.stream(prompt, max_tokens)` returns an async iterator of the generated pieces
  (`str`, never empty). **One piece is one completion token.** `max_tokens` is an int
  or `None` (no limit); the model yields at most that many pieces. It does not raise.
  `await stream.aclose()` stops the generation.
- `lm.count_tokens(text)` is the number of tokens of a prompt.

Call `lm.stream` exactly once per valid request and never for an invalid one.

**Routing**, in this order: a path other than exactly `/v1/chat/completions` gets
`404`; a method other than `POST` gets `405` with the header `allow: POST`; an invalid
body gets `400`; otherwise the response has status `200`. The query string is not part
of the path.

**The request body** is a JSON object. Fields other than these are ignored.

| field | rule |
|---|---|
| `model` | required; a non-empty string |
| `messages` | required; a non-empty array of objects, each with `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (a string, possibly empty) |
| `max_completion_tokens` | optional; an integer `>= 1` |
| `max_tokens` | optional; an integer `>= 1`. The deprecated name of the same limit: used only if `max_completion_tokens` is absent |
| `stream` | optional; a boolean, default `false` |
| `stream_options` | optional; an object whose `include_usage`, if present, is a boolean. It has an effect only when `stream` is `true` |

For the four optional fields, `null` means absent. `true` and `false` are not integers,
and `2.5` and `"3"` are not either. With neither limit, the model is called with
`max_tokens=None`.

**Errors.** Every `400` has the content type `application/json` and the body

```json
{"error": {"message": "<text>", "type": "invalid_request_error", "param": "<field>", "code": null}}
```

`message` is a non-empty string of your choice. `param` is the name of the top-level
field that is invalid (`"model"`, `"messages"`, `"max_completion_tokens"`,
`"max_tokens"`, `"stream"`, `"stream_options"`), or `null` when the body is not a JSON
object at all (not JSON, not UTF-8, empty, an array). `code` is a string or `null`. No
test request has more than one invalid field. For `404` and `405` only the status and
an `error` object with a `message` are checked.

**The prompt** is the conversation in ChatML, which `render_prompt` in the starter
produces: for each message `<|im_start|>{role}\n{content}<|im_end|>\n`, then
`<|im_start|>assistant\n`.

**Numbers.** Let $c$ be the number of pieces the model yielded.

- `finish_reason` is `"length"` if a limit was given and $c$ equals it, and `"stop"`
  otherwise. (A generation that produced exactly the limit counts as cut off, as in
  the kit's model.)
- `usage` is `{"prompt_tokens": lm.count_tokens(prompt), "completion_tokens": c,
  "total_tokens": <their sum>}`.

**Common fields.** Every response object carries `"id"`, a string that starts with
`chatcmpl-` and is different for every request; `"created"`, the Unix time in whole
seconds; and `"model"`, the request's `model`.

**Without `stream`** the response is one `application/json` object:

```json
{"id": "...", "object": "chat.completion", "created": 1794648729, "model": "...",
 "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "<the pieces joined>"},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}
```

**With `"stream": true`** the response has the content type `text/event-stream` and
consists of server-sent events `data: <payload>\n\n`. All chunks of one response have
the same `id`, `created`, and `model`, and `"object": "chat.completion.chunk"`. In
order:

1. **The role chunk**: `"choices": [{"index": 0, "delta": {"role": "assistant",
   "content": ""}, "finish_reason": null}]`.
2. **One content chunk per piece**: `"delta": {"content": "<piece>"}`,
   `"finish_reason": null`. The role is not repeated.
3. **The finish chunk**: `"delta": {}` and `"finish_reason": "stop"` or `"length"`.
4. **The usage chunk**, only if `stream_options.include_usage` is `true`:
   `"choices": []` and `"usage": {...}`. In that case every earlier chunk carries
   `"usage": null`. Otherwise there is no such chunk, and no chunk has a `usage` other
   than `null`.
5. **The terminator**: the literal `data: [DONE]`, which is not JSON, and then the end
   of the response.

An empty generation ($c = 0$) still has the role chunk and the finish chunk.

**Flushing.** Each chunk is sent in its own `http.response.body` message as soon as
its piece is available, before the next piece arrives.

**Client disconnect.** If the client closes the connection during a stream, the app
stops and closes the model's stream (`await stream.aclose()`). As in
`sse-token-stream`, `send` does not fail on a closed connection with this server; the
disconnect is announced by `receive()` returning `{"type": "http.disconnect"}`. Requests
that are not streamed are not disconnected in the tests.

Additional fields in the objects (`"logprobs": null`, `"system_fingerprint"`, and so
on) are allowed.

## Example

The model's script is `"Hello"`, `","`, `" world"`, `"!"`.

```
POST /v1/chat/completions
{"model": "tiny-chat",
 "messages": [{"role": "system", "content": "You are terse."},
              {"role": "user", "content": "Say hello"}]}
```

The prompt is

```
<|im_start|>system
You are terse.<|im_end|>
<|im_start|>user
Say hello<|im_end|>
<|im_start|>assistant
```

(with a final line feed), which `lm.count_tokens` counts as 8 tokens, and the model is
called as `lm.stream(prompt, None)`. The response:

```json
{"id": "chatcmpl-cbcd370268784e72acb31d45202c791a", "created": 1794648729,
 "model": "tiny-chat", "object": "chat.completion",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello, world!"},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}
```

The same request with `"max_tokens": 2, "stream": true, "stream_options":
{"include_usage": true}` streams (the common fields `id`, `created`, `model`, and
`object` are left out of this listing; every chunk has them):

```
data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": ","}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": null}

data: {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}

data: [DONE]

```

A body without `messages`, `{"model": "tiny-chat"}`, gets status 400 and

```json
{"error": {"message": "'messages' is required and must be a non-empty array",
           "type": "invalid_request_error", "param": "messages", "code": null}}
```

## Constraints

- Request bodies are at most 64 KiB; a generation has at most 1000 pieces.
- Responses are compared after JSON decoding: key order and spacing are free, and each
  event's JSON is on one line.
- Incremental delivery is checked by order, not by timing: the model is gated and
  produces piece `i + 1` only after the test has received the chunk for piece `i`. A
  server that holds chunks back stalls the model, which gives up after 1.5 s, and the
  case fails.
- For the disconnect check the model produces a piece every 5 ms for 200 pieces; the
  client reads four chunks and closes the connection. The case fails if the model is
  drained to its end or its stream is not closed within 5 s.
- Do not block the event loop (no `time.sleep`, no synchronous I/O).

## Why this matters

The Chat Completions format is the interface of the ecosystem: SDKs, agent
frameworks, evaluation harnesses, and gateways all speak it, so vLLM, TGI, SGLang,
llama.cpp, and Ollama each ship a server that imitates it, and a model served any
other way is a model nobody can call. The imitation has to be exact where clients
look. A `finish_reason` that always says `stop` hides truncated answers from every
caller that would have retried with a larger limit. `usage` is what billing, rate
limits, and benchmarks read, so characters instead of tokens is a wrong invoice. A
stream without `[DONE]` leaves SDKs waiting or raising on a clean ending, a role
repeated in every delta is concatenated into the message by some clients, and a
missing field answered with a 500 is counted against the server's availability
instead of being reported to the caller who made the mistake.
