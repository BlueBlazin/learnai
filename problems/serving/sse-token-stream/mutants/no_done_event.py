# BUG: the response ends after the last token without the terminal `data: [DONE]` event.
import asyncio
import json

DEFAULT_MAX_TOKENS = 16


class BadRequest(Exception):
    """The request body is not acceptable; the message says why."""


def make_app(token_source):
    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        if scope["path"] != "/v1/generate":
            await send_json(send, 404, {"error": "not found"})
            return
        if scope["method"] != "POST":
            await send_json(send, 405, {"error": "method not allowed"}, [(b"allow", b"POST")])
            return
        body = await read_body(receive)
        if body is None:
            return
        try:
            prompt, max_tokens = parse_request(body)
        except BadRequest as exc:
            await send_json(send, 400, {"error": str(exc)})
            return
        await stream_tokens(token_source(prompt, max_tokens), receive, send)

    return app


def parse_request(body: bytes) -> tuple[str, int]:
    """Return (prompt, max_tokens), or raise BadRequest saying what is wrong."""
    try:
        payload = json.loads(body)
    except ValueError:
        raise BadRequest("the body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise BadRequest("the body must be a JSON object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise BadRequest("'prompt' is required and must be a string")
    max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise BadRequest("'max_tokens' must be an integer >= 1")
    return prompt, max_tokens


async def stream_tokens(tokens, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")],
        }
    )

    async def pump():
        index = 0
        async for token in tokens:
            data = json.dumps({"token": token, "index": index})
            await send(sse_message(data, more_body=True))
            index += 1
        await send({"type": "http.response.body", "body": b"", "more_body": False})

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


def sse_message(data: str, *, more_body: bool) -> dict:
    """An http.response.body message carrying one event: a data line and a blank line."""
    return {
        "type": "http.response.body",
        "body": f"data: {data}\n\n".encode(),
        "more_body": more_body,
    }


async def read_body(receive) -> bytes | None:
    """The complete request body, or None if the client disconnected first."""
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


async def send_json(send, status: int, payload: dict, extra_headers=()) -> None:
    """Send a complete JSON response."""
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
