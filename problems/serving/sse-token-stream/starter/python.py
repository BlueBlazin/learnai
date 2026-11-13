import asyncio
import json

DEFAULT_MAX_TOKENS = 16


class BadRequest(Exception):
    """The request body is not acceptable; the message says why."""


def make_app(token_source):
    """Build the ASGI application that streams tokens from `token_source`.

    token_source: called as `token_source(prompt, max_tokens)`; returns an async
        generator that yields the generated tokens (str), one at a time, possibly with
        long pauses in between. Call it only for a valid request.
    Returns the ASGI application: an `async def app(scope, receive, send)`.
    """

    async def app(scope, receive, send):
        """Handle one connection.

        scope: dict. scope["type"] is "http" for requests; then scope["method"] is the
            upper-case HTTP method and scope["path"] the URL path without the query.
        receive: `await receive()` returns the next incoming message:
            {"type": "http.request", "body": bytes, "more_body": bool}, and after the
            whole body has been read, {"type": "http.disconnect"} once the client has
            gone away (it blocks until then).
        send: `await send(message)` with, in order:
            {"type": "http.response.start", "status": int,
             "headers": [(b"lower-case-name", b"value"), ...]}
            {"type": "http.response.body", "body": bytes, "more_body": bool}
            Every body message is written to the client immediately. The response ends
            with the first body message whose more_body is False (or missing).

        Routing: 404 for any path but /v1/generate, then 405 (with an `allow: POST`
        header) for any method but POST, then 400 for an invalid body, each with a JSON
        body {"error": "<message>"}. Otherwise stream the tokens.
        """
        if scope["type"] != "http":
            return  # "lifespan" and "websocket" scopes: nothing to do
        raise NotImplementedError

    return app


def parse_request(body: bytes) -> tuple[str, int]:
    """Validate the request body and return (prompt, max_tokens).

    The body must be a JSON object with a string "prompt" and, optionally, an integer
    "max_tokens" >= 1 (default DEFAULT_MAX_TOKENS). Raise BadRequest otherwise.
    """
    raise NotImplementedError


async def stream_tokens(tokens, receive, send) -> None:
    """Send the 200 response: one server-sent event per token, then `data: [DONE]`.

    tokens: the async generator returned by token_source.
    Each token becomes the event `data: {"token": <token>, "index": <i>}` followed by
    a blank line, sent in its own body message as soon as the token is produced. If the
    client disconnects first (receive() returns http.disconnect), stop asking for
    tokens and close the generator with `await tokens.aclose()`.
    """
    raise NotImplementedError


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
    """Send a complete JSON response with the given status code.

    extra_headers: additional (name, value) pairs of bytes, names in lower case.
    """
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
