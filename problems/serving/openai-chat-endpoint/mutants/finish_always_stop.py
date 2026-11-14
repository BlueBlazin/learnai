# BUG: finish_reason is always "stop", also when the generation was cut off by the token limit, so a client cannot tell a truncated answer from a complete one.
import asyncio
import json
import time
import uuid
from dataclasses import dataclass

PATH = "/v1/chat/completions"
ROLES = ("system", "user", "assistant")


class BadRequest(Exception):
    """The request is not acceptable. `param` names the offending field, if there is one."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    max_tokens: int | None  # None: no limit
    stream: bool
    include_usage: bool


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
        # What every object of this response shares: the chunks of one stream are
        # recognized as belonging together by their id.
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": request.model,
        }
        prompt_tokens = lm.count_tokens(prompt)
        pieces = lm.stream(prompt, request.max_tokens)
        if request.stream:
            await stream_completion(pieces, request, base, prompt_tokens, receive, send)
        else:
            await complete(pieces, request, base, prompt_tokens, send)

    return app


def parse_request(body: bytes) -> ChatRequest:
    """Validate the body and return the request, or raise BadRequest."""
    try:
        payload = json.loads(body)
    except ValueError:
        raise BadRequest("the body is not valid JSON") from None
    if not isinstance(payload, dict):
        raise BadRequest("the body must be a JSON object")

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise BadRequest("'model' is required and must be a non-empty string", "model")

    messages = payload.get("messages")  # .get: a missing field is a 400, not a KeyError
    if not isinstance(messages, list) or not messages:
        raise BadRequest("'messages' is required and must be a non-empty array", "messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ROLES
            or not isinstance(message.get("content"), str)
        ):
            raise BadRequest(
                "each message must be an object with a 'role' of system, user, or assistant "
                "and a string 'content'",
                "messages",
            )

    limits = {}
    for name in ("max_completion_tokens", "max_tokens"):
        value = payload.get(name)
        # bool is a subclass of int in Python, so `true` has to be excluded by hand.
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise BadRequest(f"'{name}' must be an integer >= 1", name)
        limits[name] = value
    max_tokens = limits["max_completion_tokens"]
    if max_tokens is None:
        max_tokens = limits["max_tokens"]  # the deprecated name, still widely sent

    stream = payload.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise BadRequest("'stream' must be a boolean", "stream")

    options = payload.get("stream_options")
    include_usage = False
    if options is not None:
        if not isinstance(options, dict) or not isinstance(options.get("include_usage", False), bool):
            raise BadRequest(
                "'stream_options' must be an object with an optional boolean 'include_usage'",
                "stream_options",
            )
        include_usage = options.get("include_usage", False)

    return ChatRequest(model, messages, max_tokens, bool(stream), include_usage)


def render_prompt(messages: list[dict]) -> str:
    """The ChatML prompt for a conversation, ending with the assistant's turn opened."""
    turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(turns) + "<|im_start|>assistant\n"


def finish_reason(completion_tokens: int, max_tokens: int | None) -> str:
    # A generation that produced exactly max_tokens pieces was cut off by the limit.
    return "stop"


def usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def complete(pieces, request, base, prompt_tokens, send) -> None:
    """The non-streaming response: one JSON object once the generation is over."""
    generated = [piece async for piece in pieces]
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "".join(generated)},
        "finish_reason": finish_reason(len(generated), request.max_tokens),
    }
    body = {
        **base,
        "object": "chat.completion",
        "choices": [choice],
        "usage": usage(prompt_tokens, len(generated)),
    }
    await send_json(send, 200, body)


async def stream_completion(pieces, request, base, prompt_tokens, receive, send) -> None:
    """The streaming response: chat.completion.chunk events, then `data: [DONE]`."""
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")],
        }
    )

    def chunk(choices, usage_value=None):
        payload = {**base, "object": "chat.completion.chunk", "choices": choices}
        if request.include_usage:
            payload["usage"] = usage_value  # null on every chunk but the last
        return sse_message(json.dumps(payload))

    def choice(delta, reason=None):
        return [{"index": 0, "delta": delta, "finish_reason": reason}]

    async def pump():
        # The role comes once, in a first chunk without text.
        await send(chunk(choice({"role": "assistant", "content": ""})))
        count = 0
        async for piece in pieces:
            # One body message per piece: it is on the wire before the next is asked for.
            await send(chunk(choice({"content": piece})))
            count += 1
        await send(chunk(choice({}, finish_reason(count, request.max_tokens))))
        if request.include_usage:
            await send(chunk([], usage(prompt_tokens, count)))
        await send(sse_message("[DONE]", more_body=False))

    async def disconnected():
        while (await receive())["type"] != "http.disconnect":
            pass

    # Sending to a closed connection does not fail, so a watcher has to listen for
    # the disconnect while the pump runs; whichever finishes first ends the request.
    tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(disconnected())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await pieces.aclose()  # stop generating for a client that has gone


def sse_message(data: str, *, more_body: bool = True) -> dict:
    """An http.response.body message carrying one server-sent event."""
    return {
        "type": "http.response.body",
        "body": f"data: {data}\n\n".encode(),
        "more_body": more_body,
    }


def error_body(message: str, param: str | None = None) -> dict:
    """An error in the shape OpenAI clients parse."""
    return {
        "error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}
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
