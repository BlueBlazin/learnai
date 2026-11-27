## The idea

The endpoint translates between HTTP/JSON and the model interface. On input, it converts a JSON conversation into a ChatML prompt string and an optional token limit. On output, it turns a stream of text pieces into either one `chat.completion` object or a sequence of `chat.completion.chunk` SSE events.

Both response modes share the same request-handling pipeline before diverging at rendering:

1. **Route and validate upfront.** Once streaming headers are sent, the HTTP status code cannot change, so every `404`, `405`, and `400` must be decided before touching the model.
2. **Fix the response identity.** Compute `id`, `created`, and `model` once per request and copy them into every output object. In a stream, the shared `id` lets clients and multiplexing proxies group chunks into one completion.
3. **Track token counts.** `prompt_tokens` is computed before generation via `lm.count_tokens(prompt)`, and `completion_tokens` is the count of yielded pieces. `finish_reason` simply compares `completion_tokens` to the token limit.
4. **Render the response.** Either collect all pieces into one JSON object or emit them incrementally as SSE chunks.

The streaming format treats each chunk as a *delta* to the final assistant message. The first delta sets the role, each content chunk appends text, and the finish chunk carries an empty delta alongside `finish_reason`. Concatenating the deltas reproduces the non-streaming message. Because token usage belongs to the whole completion rather than a single choice, it arrives in a separate chunk with `"choices": []` when `include_usage` is enabled. Finally, `data: [DONE]` is a transport-level end marker so clients can distinguish a clean finish from a dropped connection.

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

`parse_request` validates each field from the specification table using `payload.get` and raises `BadRequest(message, param)` on failure. The full reference file also includes `usage`, `sse_message`, and the starter helpers.

## Common mistakes

- **Always returning `"finish_reason": "stop"`.** Callers rely on `"length"` to detect truncated answers, such as an agent trying to parse incomplete JSON.
- **Counting characters or words instead of tokens.** `usage` must reflect the model's own tokenization, which here means `lm.count_tokens(prompt)` for prompt tokens and one token per yielded piece.
- **Omitting `[DONE]`.** Official SDKs stop iterating when they see `data: [DONE]`. Without it, clients either hang waiting for EOF or report an interrupted stream.
- **Returning `500` on missing fields.** Indexing `payload["messages"]` directly raises `KeyError` and turns a client error into a server error. Every malformed body must return `400` with the offending `param`.
- **Putting usage on the finish chunk or sending it unrequested.** Many clients index `chunk.choices[0]` on every chunk. That is why usage streaming is opt-in and why the usage chunk must be a separate event with `"choices": []`.
- **Repeating `role` in every delta.** Clients that merge deltas field by field will concatenate `"assistant"` repeatedly into the role string.
- **Generating a new `id` per chunk.** The `id` identifies the completion as a whole, not each individual SSE event.
- **Ignoring `max_completion_tokens`.** Older clients send `max_tokens` while newer ones send `max_completion_tokens`. Reading only one silently removes the token limit for the other half of callers.
- **Buffering the stream or ignoring client disconnects.** These are the same two pitfalls as in `sse-token-stream`. Because uvicorn does not raise on `send` after a disconnect, you must watch `receive()` for `http.disconnect` and close the model stream in a `finally` block.
- **Returning a flat error string.** `{"error": "message"}` breaks SDKs that expect nested `error.message`, `error.type`, `error.param`, and `error.code` fields.
- **Accepting `true` or `false` as a token limit.** In Python, `isinstance(True, int)` is `True`, so you must explicitly reject `bool` values when validating integer fields.

## Complexity

Streaming does $O(1)$ work and uses $O(1)$ extra memory per yielded piece, keeping only a running piece counter. Non-streaming buffers all pieces in memory until generation finishes, using $O(c)$ space where $c$ is the number of completion tokens.

## In the wild

- **OpenAI API specification.** The streaming chunk schema is documented in [Chat Completions streaming events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events). It specifies that `object` is always `chat.completion.chunk`, every chunk shares the same ID and timestamp, `choices` can be empty on the final chunk when `stream_options: {"include_usage": true}` is set, and `usage` is present only with that option (as `null` on all earlier chunks). The request schema in [Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) notes that `max_tokens` is deprecated in favor of `max_completion_tokens`, and defines `finish_reason` as `stop` or `length` (alongside `tool_calls` and `content_filter`, which are out of scope here). Error objects and `invalid_request_error` are described in the [error codes guide](https://developers.openai.com/api/docs/guides/error-codes).
- **Open-source serving engines.** vLLM (`vllm serve`), SGLang, TGI (Messages API), llama.cpp, and Ollama all expose this endpoint. Subtle edge-case differences still appear across servers, such as whether the first chunk includes an empty `content`, whether continuous usage stats can be attached to every chunk (vLLM's `continuous_usage_stats`), and how unknown fields are handled.
- **Chat templates.** Production servers do not hardcode ChatML. Instead, they render `messages` using the Jinja template stored in each model's tokenizer config. Applying the wrong chat template is a frequent reason a self-hosted model underperforms its published benchmarks.
- **Production additions.** Real deployments layer on authentication, `n > 1` completions, streamed tool-call arguments, stop sequences, `logprobs`, and SSE keep-alive comment lines for slow generations, while still terminating streams with `data: [DONE]`.
