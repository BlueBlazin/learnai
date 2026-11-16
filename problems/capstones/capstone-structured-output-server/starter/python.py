import asyncio
import json
import time
import uuid

from learnai.kit import DEAD, CharDFA

PATH = "/v1/chat/completions"
ROLES = ("system", "user", "assistant")


class BadRequest(Exception):
    """The request is not acceptable. `param` names the offending field, if there is one."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


def make_app(lm, vocab):
    """Build the ASGI application serving POST /v1/chat/completions.

    lm: the model (the kit's ScriptedLM).
        `lm.stream(prompt, max_tokens)` returns an async iterator of rows of logits:
        one list of `vocab.size` floats per generation step, at most `max_tokens` of
        them (None: no limit). `await stream.aclose()` stops the generation.
        `lm.count_tokens(text)` is the number of tokens in a prompt.
    vocab: the kit's ToyVocab. `vocab.bytes(id)` is a token's text as bytes (empty
        for special tokens), `vocab.eos_id` the end-of-sequence token, `vocab.size`
        the number of tokens.

    Stage 1: greedy generation, streamed and not.
    Stage 2: `response_format` with a JSON Schema constrains every token.
    Stage 3: `tools` and `tool_choice`.
    Stage 4: stop on a client disconnect; compile each schema once per app.

    Returns `async def app(scope, receive, send)`, an ASGI application. Whatever must
    live as long as the app (the compilation cache of stage 4) belongs in this closure.
    """

    async def app(scope, receive, send):
        """Route (404, then 405 with `allow: POST`), validate (400), then answer."""
        if scope["type"] != "http":
            return  # "lifespan" and "websocket" scopes: nothing to do
        raise NotImplementedError

    return app


# -- stage 2 ------------------------------------------------------------------------------


def schema_to_regex(schema: dict) -> str:
    """The pattern of the compact JSON documents that conform to `schema`.

    The schema subset and the documents are those of `json-schema-regex`, restated in
    the statement. Raises ValueError for a schema outside the subset, also when it is
    nested. The pattern must compile with `CharDFA.from_regex`.
    """
    raise NotImplementedError


def build_index(dfa: CharDFA, vocab) -> dict[int, dict[int, int]]:
    """`{state: {token: next_state}}` for every state of `dfa` and every allowed token.

    A token is allowed in a state when it has at least one byte and reading its bytes
    with `dfa.step` (or `dfa.walk`) never yields DEAD. Use `dfa.states`, `dfa.step`,
    and `dfa.walk` only.
    """
    raise NotImplementedError


# -- stage 3 ------------------------------------------------------------------------------


def parse_start():
    """The state of the tool-call parser before any text has arrived."""
    raise NotImplementedError


def parse_step(state, piece: str):
    """Consume `piece`; return `(state, events)` as in `tool-call-stream-parser`.

    Events: ("content", text), ("tool_call_start", index, name),
    ("arguments", index, text), ("tool_call_end", index). The last one has no chunk.
    """
    raise NotImplementedError


def parse_end(state) -> list[tuple]:
    """The events at the end of the text: held-back content, if any."""
    raise NotImplementedError


# -- provided helpers ---------------------------------------------------------------------


def render_prompt(messages: list[dict]) -> str:
    """The ChatML prompt for a conversation, ending with the assistant's turn opened."""
    turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(turns) + "<|im_start|>assistant\n"


def error_body(message: str, param: str | None = None) -> dict:
    """An error in the shape OpenAI clients parse."""
    return {
        "error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}
    }


def sse_message(data: str, *, more_body: bool = True) -> dict:
    """An http.response.body message carrying one server-sent event."""
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
