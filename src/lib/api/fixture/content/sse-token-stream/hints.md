A `StreamingResponse` takes an async iterator of bytes. Each item it yields is flushed to the client as its own chunk.
---
Wrap the kit generator in your own async generator that formats each token as `data: {json}\n\n` and yields `data: [DONE]\n\n` at the end.
---
For the disconnect case, iterate the kit generator inside `try` / `finally` and call `await gen.aclose()` in the `finally`. Starlette cancels your wrapper when the client goes away; the `finally` is where that cancellation becomes cleanup.
