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
    """A validated request."""

    model: str
    messages: list[dict]  # [{"role": ..., "content": ...}, ...]
    max_tokens: int | None  # the generation limit; None: no limit
    stream: bool
    include_usage: bool  # stream_options.include_usage


def make_app(lm):
    """Build the ASGI application serving POST /v1/chat/completions.

    lm: the model.
        `lm.stream(prompt, max_tokens)` returns an async iterator of the generated
        pieces (str); one piece is one completion token. max_tokens is an int or None.
        It never raises. `await stream.aclose()` stops the generation.
        `lm.count_tokens(text)` is the number of tokens in a prompt.
    Returns `async def app(scope, receive, send)`, an ASGI application (see the
    `sse-token-stream` problem for the interface).
    """

    async def app(scope, receive, send):
        """Route and validate, then answer.

        404 for another path, then 405 with an `allow: POST` header for another
        method, then 400 for an invalid body, each with `error_body(...)` as JSON.
        Otherwise call the model once, with `render_prompt(request.messages)` and
        `request.max_tokens`, and answer with `complete` or `stream_completion`.
        """
        if scope["type"] != "http":
            return  # "lifespan" and "websocket" scopes: nothing to do
        raise NotImplementedError

    return app


def parse_request(body: bytes) -> ChatRequest:
    """Validate the request body, or raise BadRequest(message, param).

    The body must be a JSON object (else param is None) with:
      model: required, a non-empty string.
      messages: required, a non-empty array of objects, each with a role in ROLES and
          a string content.
      max_completion_tokens, max_tokens: optional integers >= 1 (booleans are not
          integers). max_completion_tokens takes precedence; with neither, no limit.
      stream: optional boolean, default false.
      stream_options: optional object with an optional boolean include_usage.
    For the four optional fields, null is the same as absent. Other fields are
    ignored. `param` is the name of the top-level field that is wrong.
    """
    raise NotImplementedError


def render_prompt(messages: list[dict]) -> str:
    """The ChatML prompt for a conversation, ending with the assistant's turn opened."""
    turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(turns) + "<|im_start|>assistant\n"


async def complete(pieces, request: ChatRequest, base: dict, prompt_tokens: int, send) -> None:
    """Send the non-streaming response: status 200, one `chat.completion` JSON object.

    pieces: the async iterator from lm.stream.
    base: the fields every object of this response shares: id, created, model.
    The object has `choices` with one choice (index 0, the assistant message with the
    joined text, finish_reason "stop" or "length") and `usage` (prompt_tokens,
    completion_tokens, total_tokens).
    """
    raise NotImplementedError


async def stream_completion(
    pieces, request: ChatRequest, base: dict, prompt_tokens: int, receive, send
) -> None:
    """Send the streaming response: `chat.completion.chunk` events, then `data: [DONE]`.

    In order, one server-sent event each, sent in its own body message at once:
      1. delta {"role": "assistant", "content": ""}, finish_reason null;
      2. per piece: delta {"content": piece}, finish_reason null;
      3. delta {}, finish_reason "stop" or "length";
      4. only if request.include_usage: choices [] and the usage object (and then
         every earlier chunk carries "usage": null);
      5. the literal `data: [DONE]`.
    If the client disconnects first (receive() returns http.disconnect), stop and
    close the model's stream with `await pieces.aclose()`.
    """
    raise NotImplementedError


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
