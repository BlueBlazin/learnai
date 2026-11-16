# BUG: the cache key is the schema's JSON with sorted keys, so two schemas that declare the same properties in a different order share one automaton although their documents differ.
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import NamedTuple

from learnai.kit import DEAD, CharDFA

PATH = "/v1/chat/completions"
ROLES = ("system", "user", "assistant")
OPEN = "<tool_call>"
CLOSE = "</tool_call>"
NO_PARAMETERS = {"type": "object", "properties": {}}


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
    schema: dict | None = None  # stage 2: the schema the content must match
    parse_tools: bool = False  # stage 3: tool_choice "auto" with tools
    tool: dict | None = None  # stage 3: the function a named tool_choice selects


# -- stage 1: the generation loop ---------------------------------------------------------


class Generation:
    """One greedy generation: picks a token from every row of logits the model yields."""

    def __init__(self, rows, vocab, constraint=None):
        self.rows = rows
        self.vocab = vocab
        self.constraint = constraint
        self.count = 0  # rows consumed: the completion tokens, end-of-sequence included
        self.ended = False  # True once end-of-sequence was chosen

    async def texts(self):
        """Yield the text of each chosen token until end-of-sequence or the last row."""
        eos = self.vocab.eos_id
        constraint = self.constraint
        state = constraint.start if constraint else None
        async for row in self.rows:
            self.count += 1
            if constraint is None:
                candidates = range(len(row))
            else:
                # The index answers "which tokens keep the text inside the language";
                # ending is allowed exactly where the text so far is a whole document.
                candidates = list(constraint.moves[state])
                if state in constraint.accepting:
                    candidates.append(eos)
            token = max(candidates, key=lambda t: (row[t], -t))  # ties: the lowest id
            if token == eos:
                self.ended = True
                return
            if constraint is not None:
                state = constraint.moves[state][token]
            yield self.vocab.bytes(token).decode()


# -- stage 2: schema -> regex -> DFA -> token index -----------------------------------------

INTEGER = r"-?(0|[1-9][0-9]*)"
NUMBER = INTEGER + r"(\.[0-9]+)?([eE][+\-]?[0-9]+)?"
STRING_ITEM = r'([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'
METACHARACTERS = set(".^$*+?{}[]()|\\")


def literal(text: str) -> str:
    return "".join("\\" + char if char in METACHARACTERS else char for char in text)


def string_regex(schema: dict) -> str:
    low = schema.get("minLength", 0)
    high = schema.get("maxLength")
    if high is None:
        count = "*" if low == 0 else f"{{{low},}}"
    else:
        count = f"{{{low},{high}}}"
    return f'"{STRING_ITEM}{count}"'


def array_regex(schema: dict) -> str:
    item = schema_to_regex(schema["items"])
    low = schema.get("minItems", 0)
    high = schema.get("maxItems")
    if high == 0:
        return r"\[\]"
    more = max(low, 1) - 1
    count = f"{{{more},}}" if high is None else f"{{{more},{high - 1}}}"
    body = f"{item}(,{item}){count}"
    return rf"\[{body}\]" if low >= 1 else rf"\[({body})?\]"


def object_regex(schema: dict) -> str:
    required = set(schema.get("required", ()))
    members = [
        (literal(json.dumps(name)) + ":" + schema_to_regex(sub), name in required)
        for name, sub in schema.get("properties", {}).items()
    ]

    def rest(i: int) -> str:
        # Members i.. after some member was written: each brings its own comma.
        return "".join(f",{m}" if needed else f"(,{m})?" for m, needed in members[i:])

    def first(i: int) -> str:
        # Members i.. when nothing was written yet: the first one present has no comma.
        if i == len(members):
            return ""
        member, needed = members[i]
        present = member + rest(i + 1)
        return present if needed else f"({present}|{first(i + 1)})"

    return r"\{" + first(0) + r"\}"


def schema_to_regex(schema: dict) -> str:
    """The pattern of the compact JSON documents that conform to `schema`."""
    if not isinstance(schema, dict):
        schema = {}  # not a schema at all: rejected below like an unknown type
    if "enum" in schema:
        return "(" + "|".join(literal(json.dumps(value)) for value in schema["enum"]) + ")"
    kind = schema.get("type")
    if kind == "null":
        return "null"
    if kind == "boolean":
        return "(true|false)"
    if kind == "integer":
        return INTEGER
    if kind == "number":
        return NUMBER
    if kind == "string":
        return string_regex(schema)
    if kind == "array":
        return array_regex(schema)
    if kind == "object":
        return object_regex(schema)
    raise ValueError(f"unsupported schema type: {kind!r}")


class Constraint:
    """A compiled schema: for every DFA state, the allowed tokens and their next states."""

    def __init__(self, schema: dict, vocab):
        dfa = CharDFA.from_regex(schema_to_regex(schema))
        self.start = dfa.start
        self.accepting = dfa.accepting
        self.moves = build_index(dfa, vocab)  # every step of the DFA is taken here, once


def build_index(dfa, vocab) -> dict[int, dict[int, int]]:
    """`{state: {token: next_state}}` for the allowed tokens (see `dfa-token-index`)."""
    trie: dict = {}  # byte -> child; the key None holds the tokens that end at a node
    for token in range(vocab.size):
        piece = vocab.bytes(token)
        if not piece:
            continue  # special tokens have no text and are never allowed
        node = trie
        for byte in piece:
            node = node.setdefault(byte, {})
        node.setdefault(None, []).append(token)

    index = {}
    for state in dfa.states:
        moves: dict[int, int] = {}
        stack = [(trie, state)]
        while stack:
            node, at = stack.pop()
            for byte, child in node.items():
                if byte is None:
                    continue
                after = dfa.step(at, byte)
                if after == DEAD:
                    continue  # every token below this node dies here
                for token in child.get(None, ()):
                    moves[token] = after
                stack.append((child, after))
        index[state] = moves
    return index


def schema_key(schema: dict) -> str:
    # The text of the schema, key order kept: the order of `properties` is the order
    # of the members, so sorting the keys would merge two different languages.
    return json.dumps(schema, separators=(",", ":"), sort_keys=True)


# -- stage 3: the incremental tool-call parser (see `tool-call-stream-parser`) -------------


class ParserState(NamedTuple):
    mode: str = "content"  # "content", "head", "arguments", or "tail"
    index: int = 0  # the index of the next tool call (or of the open one)
    held: str = ""  # content: a possible beginning of OPEN; head: the string being read
    strings: int = 0  # head: string literals completed ("name", the name, "arguments")
    depth: int = 0  # arguments: open braces and brackets
    in_string: bool = False
    escaped: bool = False


def _emit(events: list, kind: str, *fields) -> None:
    *key, text = fields
    if events and events[-1][0] == kind and list(events[-1][1:-1]) == key:
        events[-1] = (kind, *key, events[-1][-1] + text)
    else:
        events.append((kind, *key, text))


def parse_step(state: ParserState, piece: str) -> tuple[ParserState, list[tuple]]:
    mode, index, held, strings, depth, in_string, escaped = state
    events: list[tuple] = []
    for ch in piece:
        if mode == "content":
            held += ch
            if held == OPEN:
                mode, held, strings = "head", "", 0
                continue
            keep = len(held)
            while keep and not OPEN.startswith(held[len(held) - keep :]):
                keep -= 1
            if keep < len(held):
                _emit(events, "content", held[: len(held) - keep])
                held = held[len(held) - keep :]
        elif mode == "head":
            if in_string:
                held += ch
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                    strings += 1
                    if strings == 2:
                        events.append(("tool_call_start", index, json.loads(held)))
            elif ch == '"':
                in_string, held = True, ch
            elif ch == "{" and strings == 3:
                mode, depth, held = "arguments", 1, ""
                _emit(events, "arguments", index, ch)
        elif mode == "arguments":
            _emit(events, "arguments", index, ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    mode, held = "tail", ""
        else:  # tail: whitespace, the brace that closes the call object, the end tag
            held += ch
            if held.endswith(CLOSE):
                mode, index, held = "content", index + 1, ""
    return ParserState(mode, index, held, strings, depth, in_string, escaped), events


def parse_end(state: ParserState) -> list[tuple]:
    if state.mode == "content" and state.held:
        return [("content", state.held)]
    return []


# -- deltas: what a token adds to the message -----------------------------------------------


def call_start(index: int, name: str) -> dict:
    """The delta that opens tool call `index`: the only one with its id, type, and name."""
    call = {
        "index": index,
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {"name": name, "arguments": ""},
    }
    return {"tool_calls": [call]}


def call_arguments(index: int, text: str) -> dict:
    return {"tool_calls": [{"index": index, "function": {"arguments": text}}]}


def event_delta(event: tuple) -> dict:
    if event[0] == "content":
        return {"content": event[1]}
    if event[0] == "tool_call_start":
        return call_start(event[1], event[2])
    return call_arguments(event[1], event[2])


async def deltas(generation: Generation, request: ChatRequest):
    """The deltas of the response, in order: the same for both response modes."""
    if request.tool is not None:
        # A named tool: the whole generation is the arguments of one call.
        yield call_start(0, request.tool["name"])
        async for text in generation.texts():
            yield call_arguments(0, text)
    elif request.parse_tools:
        state = ParserState()
        async for text in generation.texts():
            state, events = parse_step(state, text)
            for event in events:
                yield event_delta(event)
        for event in parse_end(state):
            yield event_delta(event)
    else:
        async for text in generation.texts():
            yield {"content": text}


def finish_reason(generation: Generation, called: bool) -> str:
    if not generation.ended:
        return "length"  # the rows ran out before the model chose to stop
    return "tool_calls" if called else "stop"


def usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# -- the application ------------------------------------------------------------------------


def make_app(lm, vocab):
    compiled: dict[str, Constraint] = {}  # stage 4: one compilation per schema text

    def constraint_for(schema: dict) -> Constraint:
        key = schema_key(schema)
        if key not in compiled:
            compiled[key] = Constraint(schema, vocab)
        return compiled[key]

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
            constraint = None
            if request.tool is not None:
                try:
                    constraint = constraint_for(request.tool.get("parameters", NO_PARAMETERS))
                except ValueError as exc:
                    raise BadRequest(f"unsupported tool parameters: {exc}", "tools") from None
            elif request.schema is not None:
                try:
                    constraint = constraint_for(request.schema)
                except ValueError as exc:
                    raise BadRequest(f"unsupported schema: {exc}", "response_format") from None
        except BadRequest as exc:
            await send_json(send, 400, error_body(str(exc), exc.param))
            return

        prompt = render_prompt(request.messages)
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": request.model,
        }
        prompt_tokens = lm.count_tokens(prompt)
        rows = lm.stream(prompt, request.max_tokens)
        generation = Generation(rows, vocab, constraint)
        try:
            if request.stream:
                await stream_completion(generation, request, base, prompt_tokens, receive, send)
            else:
                await complete(generation, request, base, prompt_tokens, send)
        finally:
            # Also after end-of-sequence, and for a client that has gone: stop the model.
            await rows.aclose()

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

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequest("'messages' is required and must be a non-empty array", "messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ROLES
            or not isinstance(message.get("content"), str)
        ):
            raise BadRequest(
                "each message needs a 'role' of system, user, or assistant and a string 'content'",
                "messages",
            )

    limits = {}
    for name in ("max_completion_tokens", "max_tokens"):
        value = payload.get(name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise BadRequest(f"'{name}' must be an integer >= 1", name)
        limits[name] = value
    max_tokens = limits["max_completion_tokens"]
    if max_tokens is None:
        max_tokens = limits["max_tokens"]

    stream = payload.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise BadRequest("'stream' must be a boolean", "stream")

    options = payload.get("stream_options")
    include_usage = False
    if options is not None:
        if not isinstance(options, dict) or not isinstance(
            options.get("include_usage", False), bool
        ):
            raise BadRequest("'stream_options.include_usage' must be a boolean", "stream_options")
        include_usage = options.get("include_usage", False)

    request = ChatRequest(model, messages, max_tokens, bool(stream), include_usage)
    request.schema = parse_response_format(payload.get("response_format"))
    request.parse_tools, request.tool = parse_tools(
        payload.get("tools"), payload.get("tool_choice")
    )
    return request


def parse_response_format(value) -> dict | None:
    """The schema of a `json_schema` response format; None for text."""
    if value is None:
        return None
    if isinstance(value, dict) and value.get("type") == "text":
        return None
    if isinstance(value, dict) and value.get("type") == "json_schema":
        spec = value.get("json_schema")
        if (
            isinstance(spec, dict)
            and isinstance(spec.get("name"), str)
            and spec["name"]
            and isinstance(spec.get("schema"), dict)
        ):
            return spec["schema"]
    raise BadRequest(
        "'response_format' must be {'type': 'text'} or {'type': 'json_schema', "
        "'json_schema': {'name': ..., 'schema': {...}}}",
        "response_format",
    )


def parse_tools(tools, choice) -> tuple[bool, dict | None]:
    """Return (parse tool calls in the text?, the function a named choice selects)."""
    functions = {}
    if tools is not None:
        if not isinstance(tools, list) or not tools:
            raise BadRequest("'tools' must be a non-empty array", "tools")
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if (
                not isinstance(tool, dict)
                or tool.get("type") != "function"
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
                or not isinstance(function.get("parameters", {}), dict)
            ):
                raise BadRequest(
                    "each tool must be {'type': 'function', 'function': {'name': ..., "
                    "'parameters': {...}}}",
                    "tools",
                )
            functions[function["name"]] = function

    if choice is None:
        choice = "auto" if functions else "none"
    if choice == "none":
        return False, None
    if choice == "auto" and functions:
        return True, None
    if isinstance(choice, dict) and choice.get("type") == "function":
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name in functions:
            return False, functions[name]
    raise BadRequest(
        "'tool_choice' must be 'none', 'auto' (with tools), or name one of the tools",
        "tool_choice",
    )


def render_prompt(messages: list[dict]) -> str:
    """The ChatML prompt for a conversation, ending with the assistant's turn opened."""
    turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(turns) + "<|im_start|>assistant\n"


def has_call(delta: dict) -> bool:
    return "tool_calls" in delta


def apply_delta(message: dict, delta: dict) -> None:
    """Add a delta to the message it belongs to (what a streaming client does)."""
    if "content" in delta:
        message["content"] += delta["content"]
    for part in delta.get("tool_calls", ()):
        calls = message.setdefault("tool_calls", [])
        if "id" in part:
            calls.append({"id": part["id"], "type": "function", "function": dict(part["function"])})
        else:
            calls[part["index"]]["function"]["arguments"] += part["function"]["arguments"]


async def complete(generation, request, base, prompt_tokens, send) -> None:
    """The non-streaming response: the message that the deltas add up to."""
    message = {"role": "assistant", "content": ""}
    async for delta in deltas(generation, request):
        apply_delta(message, delta)
    called = "tool_calls" in message
    if called and not message["content"]:
        message["content"] = None
    choice = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason(generation, called),
    }
    body = {
        **base,
        "object": "chat.completion",
        "choices": [choice],
        "usage": usage(prompt_tokens, generation.count),
    }
    await send_json(send, 200, body)


async def stream_completion(generation, request, base, prompt_tokens, receive, send) -> None:
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
            payload["usage"] = usage_value
        return sse_message(json.dumps(payload))

    def choice(delta, reason=None):
        return [{"index": 0, "delta": delta, "finish_reason": reason}]

    async def pump():
        await send(chunk(choice({"role": "assistant", "content": ""})))
        called = False
        async for delta in deltas(generation, request):
            called = called or has_call(delta)
            await send(chunk(choice(delta)))
        await send(chunk(choice({}, finish_reason(generation, called))))
        if request.include_usage:
            await send(chunk([], usage(prompt_tokens, generation.count)))
        await send(sse_message("[DONE]", more_body=False))

    async def disconnected():
        while (await receive())["type"] != "http.disconnect":
            pass

    # Sending to a closed connection does not fail, so a watcher listens for the
    # disconnect while the pump runs; whichever finishes first ends the request.
    tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(disconnected())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def sse_message(data: str, *, more_body: bool = True) -> dict:
    return {
        "type": "http.response.body",
        "body": f"data: {data}\n\n".encode(),
        "more_body": more_body,
    }


def error_body(message: str, param: str | None = None) -> dict:
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
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
