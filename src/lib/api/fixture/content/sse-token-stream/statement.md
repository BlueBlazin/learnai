Users judge a model server by how soon the first token appears. Streaming sends each token as it is generated, over a single HTTP response, using **server-sent events**.

Build an ASGI app with one endpoint that streams tokens from a generator supplied by the kit.

## Wire format

A server-sent event is a block of `field: value` lines ended by a blank line. Your stream must look exactly like this:

```
data: {"index": 0, "token": "The"}

data: {"index": 1, "token": " quick"}

data: [DONE]

```

## Endpoint

| | |
|---|---|
| Method and path | `POST /v1/stream` |
| Request body | `{"prompt": string, "max_tokens": integer}` |
| Response status | `200` |
| Response headers | `Content-Type: text/event-stream`, `Cache-Control: no-cache` |
| Errors | `422` with `{"error": string}` when the body is invalid |

## Interface

```python
def create_app(generate: Callable[[str, int], AsyncIterator[str]]) -> ASGIApp:
    """Return an ASGI app. `generate(prompt, max_tokens)` yields token strings."""
```

## How it is judged

The service judge starts your app on an ephemeral local port and connects with a real HTTP client.

1. **Format.** Events parse, indices are consecutive from $0$, and the stream ends with `[DONE]`.
2. **Incremental delivery.** The kit generator pauses between tokens. The first event must reach the client before the generator has produced the third token; a handler that buffers the whole response fails.
3. **Disconnect.** When the client closes the connection early, the generator must be closed within 100 ms. The judge checks that its `finally` block ran.
4. **Validation.** A missing `prompt` or a non-positive `max_tokens` returns `422` and never calls the generator.

> [!NOTE]
> You may use Starlette (`StreamingResponse`) or write raw ASGI. Both are installed. Raw ASGI is about forty lines and teaches more.

## Constraints

- $1 \le$ `max_tokens` $\le 4096$.
- Tokens may contain newlines and quotes; encode each event's payload with `json.dumps`.
