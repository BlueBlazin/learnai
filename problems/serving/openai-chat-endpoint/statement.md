Implement `POST /v1/chat/completions` following OpenAI's Chat Completions API, supporting both single JSON responses and streamed Server-Sent Events (SSE). The model handles text generation, while your endpoint validates requests, tracks token usage and finish reasons, formats streaming deltas, and closes generation cleanly if a client disconnects.

![This problem is the Endpoint, which speaks the Chat Completions protocol.](assets/map.svg)

## Specification

```python
def make_app(lm):
    async def app(scope, receive, send): ...
    return app
```

`make_app` returns an ASGI application. The test suite serves it with uvicorn on a local socket and sends HTTP requests to it. The starter code uses the raw ASGI interface from [`sse-token-stream`](../sse-token-stream/statement.md) and provides `read_body`, `send_json`, `render_prompt`, and `error_body`. You may use Starlette instead if you prefer.

**The model.** `lm` is a [`ScriptedLM`](../../../python/src/learnai/kit/scriptedlm.py) instance with two methods:

- `lm.stream(prompt, max_tokens)` returns an async iterator of generated text pieces (`str`, never empty). **Each piece counts as one completion token.** `max_tokens` is an `int` or `None` (no limit), and the model yields at most that many pieces without raising. Calling `await stream.aclose()` stops generation early.
- `lm.count_tokens(text)` returns the integer token count of a prompt string.

Call `lm.stream` exactly once per valid request and never for an invalid one.

**Routing.** Check requests in this order (ignoring any query string):

1. Any path other than `/v1/chat/completions` returns `404`.
2. Any method other than `POST` returns `405` with the header `allow: POST`.
3. An invalid request body returns `400`.
4. Otherwise, return `200`.

**The request body.** The body must be a JSON object. Ignore any fields not listed below.

| Field | Rule |
|---|---|
| `model` | Required. A non-empty string. |
| `messages` | Required. A non-empty array of objects, each with `role` (`"system"`, `"user"`, or `"assistant"`) and `content` (a string, possibly empty). |
| `max_completion_tokens` | Optional. An integer `>= 1`. |
| `max_tokens` | Optional. An integer `>= 1`. Deprecated alias for the token limit, used only when `max_completion_tokens` is absent. |
| `stream` | Optional. A boolean, defaulting to `false`. |
| `stream_options` | Optional. An object whose `include_usage` field, if present, is a boolean. Applies only when `stream` is `true`. |

For the four optional fields, `null` counts as absent. Booleans (`true`, `false`), floats (`2.5`), and strings (`"3"`) are not valid integers. If neither token limit is provided, pass `max_tokens=None` to `lm.stream`.

**Errors.** Every `400` response uses `content-type: application/json` and has this shape:

```json
{"error": {"message": "<text>", "type": "invalid_request_error", "param": "<field>", "code": null}}
```

- `message` is a non-empty string describing the problem.
- `param` is the name of the invalid top-level field (`"model"`, `"messages"`, `"max_completion_tokens"`, `"max_tokens"`, `"stream"`, or `"stream_options"`), or `null` if the body is not a valid JSON object (for example, malformed JSON, invalid UTF-8, empty body, or a JSON array).
- `code` is a string or `null`.

No test request contains more than one invalid field. For `404` and `405` responses, tests check only the status code (plus the `allow` header for `405`) and an `error` object containing a `message`.

**The prompt.** Format the conversation in ChatML, which `render_prompt` in the starter code produces. Each message becomes `<|im_start|>{role}\n{content}<|im_end|>\n`, followed by `<|im_start|>assistant\n`.

**Token counts and finish reason.** Let $c$ be the number of pieces yielded by `lm.stream`.

- `finish_reason` is `"length"` if a token limit was set and $c$ equals that limit, and `"stop"` otherwise. Reaching the limit exactly counts as truncated.
- `usage` is `{"prompt_tokens": lm.count_tokens(prompt), "completion_tokens": c, "total_tokens": prompt_tokens + c}`.

**Common response fields.** Every response object includes three shared fields:

- `"id"`: a string starting with `chatcmpl-` that is unique per request.
- `"created"`: the current Unix timestamp in whole seconds.
- `"model"`: the `model` string from the request.

**Non-streaming responses (`stream` is `false` or absent).** Return a single `application/json` object:

```json
{"id": "...", "object": "chat.completion", "created": 1794648729, "model": "...",
 "choices": [{"index": 0,
              "message": {"role": "assistant", "content": "<the pieces joined>"},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}
```

**Streaming responses (`stream` is `true`).** Return `content-type: text/event-stream` as a sequence of SSE events formatted as `data: <payload>\n\n`. Every JSON chunk in a response shares the same `id`, `created`, and `model`, and sets `"object": "chat.completion.chunk"`. Send events in this order:

1. **Role chunk.** Set `"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]`.
2. **Content chunks.** Send one chunk per yielded piece with `"delta": {"content": "<piece>"}` and `"finish_reason": null`. Do not repeat `role`.
3. **Finish chunk.** Set `"delta": {}` and `"finish_reason"` to `"stop"` or `"length"`.
4. **Usage chunk (optional).** If `stream_options.include_usage` is `true`, send a chunk with `"choices": []` and `"usage": {...}`, and include `"usage": null` on every earlier chunk. Otherwise, omit this chunk and never set `usage` to a non-null value.
5. **Terminator.** Send the literal event `data: [DONE]\n\n` (not JSON) and end the response.

An empty generation ($c = 0$) still emits the role chunk and the finish chunk (plus the usage chunk if requested, and `[DONE]`).

**Flushing.** Send each chunk in its own `http.response.body` message as soon as its piece arrives, without waiting for the next piece.

**Client disconnect.** If the client closes the connection mid-stream, stop streaming and call `await stream.aclose()`. Under uvicorn, `send` does not raise on a closed connection, so detect disconnects by watching `receive()` for `{"type": "http.disconnect"}` (as in `sse-token-stream`). Tests do not disconnect non-streaming requests.

Extra optional fields (such as `"logprobs": null` or `"system_fingerprint"`) are allowed.

## Example

Suppose the model is scripted to yield `"Hello"`, `","`, `" world"`, and `"!"`.

Consider this non-streaming request:

```
POST /v1/chat/completions
{"model": "tiny-chat",
 "messages": [{"role": "system", "content": "You are terse."},
              {"role": "user", "content": "Say hello"}]}
```

`render_prompt` formats the messages into the following prompt (ending with a newline):

```
<|im_start|>system
You are terse.<|im_end|>
<|im_start|>user
Say hello<|im_end|>
<|im_start|>assistant
```

Here `lm.count_tokens(prompt)` returns `8`, and the app calls `lm.stream(prompt, None)`. The model yields all 4 pieces, producing this JSON response:

```json
{"id": "chatcmpl-cbcd370268784e72acb31d45202c791a", "created": 1794648729,
 "model": "tiny-chat", "object": "chat.completion",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello, world!"},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}}
```

Now add `"max_tokens": 2, "stream": true, "stream_options": {"include_usage": true}` to the same request. The app calls `lm.stream(prompt, 2)` and streams the events below (omitting the common fields `id`, `created`, `model`, and `object` from each line for readability, though every JSON chunk must include them):

```
data: {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {"content": ","}, "finish_reason": null}], "usage": null}

data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": null}

data: {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}}

data: [DONE]

```

Finally, a request missing `messages`, such as `{"model": "tiny-chat"}`, returns status `400`:

```json
{"error": {"message": "'messages' is required and must be a non-empty array",
           "type": "invalid_request_error", "param": "messages", "code": null}}
```

## Constraints

- Request bodies are at most 64 KiB, and each generation yields at most 1000 pieces.
- Tests decode JSON before comparing responses, so key order and whitespace do not matter, provided each SSE event's JSON is on a single line.
- Incremental delivery is tested by gating rather than wall-clock timing. The model produces piece `i + 1` only after the client receives the chunk for piece `i`. If your server buffers chunks, the model times out after 1.5 s and the test fails.
- In the disconnect test, the model yields one piece every 5 ms for 200 pieces while the client reads four chunks and disconnects. The test fails if the model runs to completion or if its stream is not closed within 5 s.
- Do not block the event loop (no `time.sleep` or synchronous I/O).

## Why this matters

OpenAI's Chat Completions format is the standard interface across LLM SDKs, agent frameworks, evaluation harnesses, and gateways. Inference servers like vLLM, SGLang, TGI, llama.cpp, and Ollama all implement it so existing clients can talk to any model without code changes.

Small protocol mistakes cause silent failures in real clients. Always returning `"finish_reason": "stop"` hides truncated outputs from callers that retry with a larger token budget. Reporting character counts instead of token counts corrupts billing, rate limiting, and benchmarks. Omitting `[DONE]` leaves SDKs hanging or raising stream-interruption errors, repeating `role` in every delta breaks clients that concatenate delta fields, and returning `500` instead of `400` on invalid input counts caller bugs against server availability.
