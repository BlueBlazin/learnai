An HTTP response does not have to be sent in one piece. In ASGI, what marks the end of
the response, and what happens on the wire each time you `await send(...)` a body
message that is not the last?

---

After validating the request, send `http.response.start` with status 200 and the
`text/event-stream` content type. Then loop over the generator with `async for`: for
each token, build `data: ` + `json.dumps({"token": token, "index": i})` + two
newlines, encode it as UTF-8, and send it right away as an `http.response.body` message
with `"more_body": True`. After the loop, send `data: [DONE]` plus two newlines with
`"more_body": False`. Do all validation (path, method, body) before the first `send`:
once the 200 is out you cannot turn it into a 400. Mind that `isinstance(True, int)` is
true in Python.

---

`send` will not tell you the client has left, and the token loop is stuck in
`async for` while the model thinks, so something else has to listen. Put the token
loop in one coroutine and a disconnect watcher in another, and race them:

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

Whichever finishes first ends the request: the pump after `[DONE]`, or the watcher
when the client hangs up. Canceling the pump interrupts it even in the middle of
waiting for the next token.
