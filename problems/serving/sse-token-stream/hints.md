An HTTP response does not have to be sent all at once. In ASGI, what field marks the final body message, and what does the server write to the socket each time you `await send(...)` a body message with `more_body` set to `True`?

---

Validate the path, method, and JSON body before calling `send()` for the first time, because once `200` is sent you cannot change it to a `400`. Remember that `isinstance(True, int)` is `True` in Python, so reject booleans explicitly when checking `max_tokens`.

For a valid request, send `http.response.start` with status `200` and content type `text/event-stream`. Then iterate over the generator with `async for`. For each token, format `f"data: {json.dumps({'token': token, 'index': i})}\n\n"`, encode it as UTF-8, and send it immediately as an `http.response.body` message with `"more_body": True`. After the loop ends, send `b"data: [DONE]\n\n"` with `"more_body": False`.

---

`send()` will not reliably tell you when the client hangs up, and the token loop stays blocked in `async for` while waiting for the next token. Run the token loop and a disconnect listener as two concurrent tasks, and stop as soon as either finishes:

```python
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
```

Whichever task finishes first ends the stream, whether the pump finishes after `[DONE]` or the listener sees `http.disconnect`. Canceling the pump interrupts it even while it is awaiting the next token, and `await tokens.aclose()` ensures the generator is closed cleanly.
