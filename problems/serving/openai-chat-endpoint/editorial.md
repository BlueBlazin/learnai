## The idea

The endpoint is a translation layer. On the way in, a JSON conversation becomes a
prompt string and a token limit. On the way out, a sequence of pieces becomes either
one `chat.completion` object or a sequence of `chat.completion.chunk` events. Nothing
in between is specific to the mode, so the solution has one spine and two renderers:

1. **Route and validate** before anything is sent. The status line cannot be changed
   once the stream has started, so every 4xx is decided up front.
2. **Fix the response's identity**: `id`, `created`, and `model` are computed once and
   copied into every object. In a stream the `id` is what tells a client (or a proxy
   that multiplexes streams) that chunks belong together.
3. **Count.** The usage is two integers: the prompt's token count, known before
   generation, and the number of pieces, known after. The finish reason is a
   comparison of the second with the limit.
4. **Render**, as one object or as events.

The streaming format follows from one design decision of the API: a chunk is a
*delta* to the message that the non-streaming response would contain. The first delta
sets the role, each further delta appends to the content, the finish chunk has nothing
left to add and sets the choice's `finish_reason`, and concatenating the deltas
reproduces the message. The usage does not belong to any choice, so when the client
asks for it, it travels in a chunk of its own whose `choices` list is empty. `[DONE]`
is not JSON and not a chunk: it is the transport's end marker, the only way a client
can tell a finished stream from a dropped connection.

## Reference solution (Python)

```python
def make_app(lm):
    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        if scope["path"] != PATH:
            await send_json(send, 404, error_body("not found"))
            return
        if scope["method"] != "POST":
            await send_json(send, 405, error_body("method not allowed"), [(b"allow", b"POST")])
            return
        body = await read_body(receive)
        if body is None:
            return
        try:
            request = parse_request(body)
        except BadRequest as exc:
            await send_json(send, 400, error_body(str(exc), exc.param))
            return

        prompt = render_prompt(request.messages)
        base = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()),
                "model": request.model}
        prompt_tokens = lm.count_tokens(prompt)
        pieces = lm.stream(prompt, request.max_tokens)
        if request.stream:
            await stream_completion(pieces, request, base, prompt_tokens, receive, send)
        else:
            await complete(pieces, request, base, prompt_tokens, send)

    return app


def finish_reason(completion_tokens, max_tokens):
    return "length" if completion_tokens == max_tokens else "stop"


async def complete(pieces, request, base, prompt_tokens, send):
    generated = [piece async for piece in pieces]
    choice = {"index": 0,
              "message": {"role": "assistant", "content": "".join(generated)},
              "finish_reason": finish_reason(len(generated), request.max_tokens)}
    await send_json(send, 200, {**base, "object": "chat.completion", "choices": [choice],
                                "usage": usage(prompt_tokens, len(generated))})


async def stream_completion(pieces, request, base, prompt_tokens, receive, send):
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"text/event-stream"),
                            (b"cache-control", b"no-cache")]})

    def chunk(choices, usage_value=None):
        payload = {**base, "object": "chat.completion.chunk", "choices": choices}
        if request.include_usage:
            payload["usage"] = usage_value
        return sse_message(json.dumps(payload))

    def choice(delta, reason=None):
        return [{"index": 0, "delta": delta, "finish_reason": reason}]

    async def pump():
        await send(chunk(choice({"role": "assistant", "content": ""})))
        count = 0
        async for piece in pieces:
            await send(chunk(choice({"content": piece})))
            count += 1
        await send(chunk(choice({}, finish_reason(count, request.max_tokens))))
        if request.include_usage:
            await send(chunk([], usage(prompt_tokens, count)))
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
        await pieces.aclose()
```

`parse_request` checks the fields in the statement's table one by one with
`payload.get`, raising `BadRequest(message, param)`; the full file has it, together
with `usage`, `sse_message`, and the starter's helpers.

## Common mistakes

- **`finish_reason` always `"stop"`.** The one field that tells a caller its answer
  was truncated. Agents that parse JSON out of the answer depend on it to know why
  the JSON is incomplete.
- **Usage in characters**, or in words of the output, or from a different tokenizer
  than the model's. Usage is the model's own count: here `lm.count_tokens` for the
  prompt and one per piece.
- **No `[DONE]`.** The official SDKs stop iterating at `[DONE]`; without it some
  clients wait for the connection to close and others report an interrupted stream.
- **500 on a missing field.** `payload["messages"]` raises `KeyError`, the server
  answers 500, and the caller's bug is logged as the server's. Every way a body can be
  wrong must end in a 400 that names the field.
- **Usage on the finish chunk** instead of in a separate chunk with `"choices": []`,
  or a usage chunk nobody asked for. Clients that index `chunk.choices[0]` on every
  chunk are the reason the API made this opt-in, and the reason the usage chunk must
  be exactly where the option says.
- **The role in every delta.** Deltas are concatenated; a client that merges them
  field by field ends up with the role `assistantassistantassistant`.
- **A new `id` per chunk.** The id identifies the completion, not the event.
- **Ignoring `max_completion_tokens`.** Older clients send `max_tokens`, newer ones
  the new name; a server that reads only one of them silently generates without a
  limit for half its callers.
- **Buffering the stream**, and **ignoring the disconnect**: the two failures of
  `sse-token-stream`, with the same fixes. A send to a closed connection does not
  raise under uvicorn, so only `receive()` reveals that the client has gone.
- **A flat error.** `{"error": "message"}` is not what clients parse; the SDKs read
  `error.message`, `error.type`, `error.param`, and `error.code`.
- **`true` as a token limit.** `isinstance(True, int)` holds in Python.

## Complexity

Constant work per piece. The streaming path keeps a counter and nothing else; the
non-streaming path holds the pieces until the end, which is inherent in the format.

## In the wild

- **The specification** is OpenAI's API reference. The chunk object is described
  under [Chat Completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events):
  `object` "is always `chat.completion.chunk`", each chunk "has the same ID" and "the
  same timestamp", `choices` "can also be empty for the last chunk if you set
  `stream_options: {"include_usage": true}`", and `usage` is "an optional field that
  will only be present" with that option and "contains a null value except for the
  last chunk". The request side is under
  [Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create):
  with `include_usage`, "an additional chunk will be streamed before the `data:
  [DONE]` message", and `max_tokens` "is now deprecated in favor of
  `max_completion_tokens`". `finish_reason` is `stop` for "a natural stop point or a
  provided stop sequence" and `length` "if the maximum number of tokens specified in
  the request was reached"; the other values (`tool_calls`, `content_filter`) are
  outside this problem. Errors are an `error` object whose `type`, `param`, and `code`
  the [error codes guide](https://developers.openai.com/api/docs/guides/error-codes)
  refers to, with `invalid_request_error` for a bad parameter.
- **vLLM** (`vllm serve`), **SGLang**, **TGI** (its Messages API), **llama.cpp**'s
  server, and **Ollama** all implement this endpoint. They differ at the edges this
  problem pins down: whether the first chunk carries an empty `content`, whether
  usage can also be attached to every chunk (vLLM's `continuous_usage_stats`), and
  how unsupported fields are treated. Testing a client against two servers finds
  these quickly.
- **The chat template** is the model's, not the server's: real servers render
  `messages` with the Jinja template stored in the tokenizer's configuration, of
  which the ChatML used here is one common instance. A wrong template is the
  classic cause of a served model that is "worse than the paper".
- **What production adds**: authentication, `n > 1`, tool calls (their arguments are
  streamed as deltas too), stop sequences, `logprobs`, and a keep-alive comment line
  for slow streams. The stream still ends with `data: [DONE]`.
