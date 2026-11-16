"""Tests for capstone-structured-output-server.

Every case builds the app with `make_app(lm, vocab)`, serves it on a local socket, and
talks to it over HTTP. `lm` is the kit's ScriptedLM whose script is a list of logits
rows (one list of `vocab.size` floats per generation step), and `vocab` a ToyVocab with
282 tokens, so every byte has a token.

What a response must contain is computed here, not by the reference server: `greedy`
replays the model's rows (with the kit's DFAConstraint when the request has a schema),
and `expect` turns the chosen tokens into the deltas of the response. Two pieces of the
reference module are used as oracles: `schema_to_regex` (a schema's pattern) and
`parse_step` / `parse_end` (the events of the tool-call markup). Constrained output is
also checked directly against the schema by `conforms`, which knows nothing of regexes.

Stage 4 counts compilations by wrapping `CharDFA.from_regex` for the length of a case:
every automaton the server compiles is recorded, and its step counter read.
"""

import json

import httpx

from whetstone.judge import CheckFailure, case, check, iter_sse
from whetstone.kit import CharDFA, DFAConstraint, ScriptedLM, ToyVocab

S1 = "1: chat completions"
S2 = "2: schema-constrained output"
S3 = "3: tool calls"
S4 = "4: cancellation and the compilation cache"

URL = "/v1/chat/completions"
CONVERSATION = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Say hello"},
]
EOS = None  # in a ranking: the end-of-sequence token

# The compiler the tests use for their own automata; stage 4 wraps the class attribute.
_FROM_REGEX = CharDFA.__dict__["from_regex"].__func__


_DFAS = {}


def compile_dfa(pattern):
    if pattern not in _DFAS:
        _DFAS[pattern] = _FROM_REGEX(CharDFA, pattern)
    return _DFAS[pattern]


# -- the model's rows ---------------------------------------------------------------------

VOCAB = ToyVocab(282, seed=0)
ASCII = [t for t, piece in enumerate(VOCAB.pieces) if piece and all(b < 128 for b in piece)]


def token(piece):
    if piece is EOS:
        return VOCAB.eos_id
    return VOCAB.id_of(piece.encode() if isinstance(piece, str) else piece)


def base_row():
    """Logits no test wants chosen: ASCII tokens in id order, far above everything else."""
    row = [-100.0 - t for t in range(VOCAB.size)]
    for t in ASCII:
        row[t] = -1.0 - t / 1000
    return row


def ranked(*pieces):
    """A row that prefers the given tokens in this order (text, bytes, or EOS)."""
    row = base_row()
    for rank, piece in reversed(list(enumerate(pieces))):
        row[token(piece)] = 10.0 - rank
    return row


def spell(text, *then):
    """Rows whose top choices spell `text`; `then` are every row's next preferences."""
    return [ranked(VOCAB.bytes(t), *then) for t in VOCAB.encode(text)]


def random_rows(rng, n, favorites, eos_from=None):
    """Seeded rows: noise on the ASCII tokens, a bonus for `favorites`, EOS late."""
    favorite_ids = [token(piece) for piece in favorites]
    rows = []
    for i in range(n):
        row = [-100.0 - t for t in range(VOCAB.size)]
        for t in ASCII:
            row[t] = float(rng.normal())
        for t in favorite_ids:
            row[t] += float(rng.uniform(0.0, 4.0))
        if eos_from is not None and i >= eos_from:
            row[VOCAB.eos_id] = 50.0
        rows.append(row)
    return rows


# -- what the server must produce ---------------------------------------------------------


def greedy(rows, limit, dfa=None):
    """Replay the rows: (texts of the chosen tokens, rows consumed, EOS chosen?)."""
    constraint = DFAConstraint(dfa, VOCAB) if dfa is not None else None
    state = constraint.start() if constraint else None
    texts, count = [], 0
    for row in rows if limit is None else rows[:limit]:
        count += 1
        if constraint is None:
            candidates = list(range(VOCAB.size))
        else:
            candidates = list(constraint.allowed(state))
            if constraint.is_accepting(state):
                candidates.append(VOCAB.eos_id)
        choice = max(candidates, key=lambda t: (row[t], -t))
        if choice == VOCAB.eos_id:
            return texts, count, True
        if constraint is not None:
            state = constraint.advance(state, choice)
        texts.append(VOCAB.pieces[choice].decode())
    return texts, count, False


def limit_of(body):
    limit = body.get("max_completion_tokens")
    return body.get("max_tokens") if limit is None else limit


def render_prompt(messages):
    turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(turns) + "<|im_start|>assistant\n"


class Expected:
    """The deltas, finish reason, and usage of one response."""

    def __init__(self, deltas, count, ended, prompt):
        self.deltas = deltas  # tool-call ids are the placeholder "call_#"
        self.count = count
        called = any("tool_calls" in delta for delta in deltas)
        self.finish = "length" if not ended else ("tool_calls" if called else "stop")
        prompt_tokens = len(prompt.split())
        self.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": count,
            "total_tokens": prompt_tokens + count,
        }
        message = {"role": "assistant", "content": ""}
        for delta in deltas:
            if "content" in delta:
                message["content"] += delta["content"]
            for part in delta.get("tool_calls", ()):
                calls = message.setdefault("tool_calls", [])
                if "id" in part:
                    calls.append(
                        {"id": "call_#", "type": "function", "function": dict(part["function"])}
                    )
                else:
                    calls[part["index"]]["function"]["arguments"] += part["function"]["arguments"]
        if called and not message["content"]:
            message["content"] = None
        self.message = message

    @property
    def text(self):
        return self.message["content"]


def start_delta(index, name):
    call = {
        "index": index,
        "id": "call_#",
        "type": "function",
        "function": {"name": name, "arguments": ""},
    }
    return {"tool_calls": [call]}


def arguments_delta(index, text):
    return {"tool_calls": [{"index": index, "function": {"arguments": text}}]}


def expect(ctx, body, rows):
    """What the statement's rules give for this request and these rows."""
    prompt = render_prompt(body["messages"])
    tools = {tool["function"]["name"]: tool["function"] for tool in body.get("tools") or ()}
    choice = body.get("tool_choice") or ("auto" if tools else "none")
    spec = body.get("response_format") or {"type": "text"}
    if isinstance(choice, dict):
        function = tools[choice["function"]["name"]]
        schema = function.get("parameters", {"type": "object", "properties": {}})
        dfa = compile_dfa(ctx.oracle.schema_to_regex(schema))
        texts, count, ended = greedy(rows, limit_of(body), dfa)
        deltas = [start_delta(0, function["name"])]
        deltas += [arguments_delta(0, text) for text in texts]
    elif choice == "auto":
        texts, count, ended = greedy(rows, limit_of(body))
        deltas = []
        state = ctx.oracle.ParserState()
        events = []
        for text in texts:
            state, new = ctx.oracle.parse_step(state, text)
            events += new
        events += ctx.oracle.parse_end(state)
        for event in events:
            if event[0] == "content":
                deltas.append({"content": event[1]})
            elif event[0] == "tool_call_start":
                deltas.append(start_delta(event[1], event[2]))
            elif event[0] == "arguments":
                deltas.append(arguments_delta(event[1], event[2]))
    else:
        dfa = None
        if spec["type"] == "json_schema":
            dfa = compile_dfa(ctx.oracle.schema_to_regex(spec["json_schema"]["schema"]))
        texts, count, ended = greedy(rows, limit_of(body), dfa)
        deltas = [{"content": text} for text in texts]
    return Expected(deltas, count, ended, prompt)


def conforms(schema, value):
    """Does a parsed JSON value conform to a schema of the statement's subset?"""
    if "enum" in schema:
        return any(type(value) is type(v) and value == v for v in schema["enum"])
    kind = schema["type"]
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        if not isinstance(value, list):
            return False
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            return False
        return all(conforms(schema["items"], item) for item in value)
    if not isinstance(value, dict):
        return False
    properties = schema.get("properties", {})
    order = [name for name in properties if name in value]
    if list(value) != order or any(name not in value for name in schema.get("required", ())):
        return False
    return all(conforms(properties[name], item) for name, item in value.items())


def check_document(schema, text, label, detail=""):
    """A complete constrained output: JSON, conforming, and compact."""
    detail += f"schema: {json.dumps(schema)}\noutput: {text!r}\n"
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise CheckFailure(f"{label}: the output is not valid JSON", detail) from None
    check.that(
        conforms(schema, value), f"{label}: the output does not conform to the schema", detail
    )
    check.that(
        not whitespace_outside_strings(text), f"{label}: the output is not compact JSON", detail
    )


def whitespace_outside_strings(text):
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in " \t\r\n":
            return True
    return False


# -- talking to the server ----------------------------------------------------------------


def body_of(messages=CONVERSATION, model="tiny-chat", **fields):
    return {"model": model, "messages": messages, **fields}


def schema_format(schema, name="answer"):
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


def tool(name, parameters=None, description=None):
    function = {"name": name}
    if description is not None:
        function["description"] = description
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def named(name):
    return {"type": "function", "function": {"name": name}}


def serve(ctx, lm):
    return ctx.serve(ctx.solution.make_app(lm, VOCAB))


def send(client, method, path, label, **kwargs):
    try:
        return client.request(method, path, **kwargs)
    except httpx.TimeoutException:
        raise CheckFailure(f"{label}: timed out waiting for the complete response") from None
    except httpx.HTTPError as exc:
        raise CheckFailure(
            f"{label}: the connection broke: {type(exc).__name__}", f"{exc}\n"
        ) from None


def describe(label, body):
    return f"request: {label}\nbody: {json.dumps(body)[:500]}\n"


def check_base(obj, object_type, model, label, detail):
    check.that(isinstance(obj, dict), f"{label}: the response is not a JSON object", detail)
    check.that(
        obj.get("object") == object_type, f'{label}: "object" must be "{object_type}"', detail
    )
    ident = obj.get("id")
    check.that(
        isinstance(ident, str) and ident.startswith("chatcmpl-") and len(ident) > len("chatcmpl-"),
        f'{label}: "id" must be a string that starts with "chatcmpl-"',
        detail,
    )
    created = obj.get("created")
    check.that(
        isinstance(created, int) and not isinstance(created, bool) and created > 0,
        f'{label}: "created" must be a Unix timestamp in whole seconds',
        detail,
    )
    check.that(obj.get("model") == model, f'{label}: "model" must echo the request', detail)


def check_model_call(lm, index, body, want, label, detail):
    """One more call, with the ChatML prompt and the limit; stopped where it should."""
    check.that(
        len(lm.calls) == index + 1,
        f"{label}: the model must be called exactly once per valid request",
        detail + f"lm.stream was called {len(lm.calls)} times in total, expected {index + 1}\n",
    )
    call = lm.calls[index]
    prompt = render_prompt(body["messages"])
    check.that(
        call.prompt == prompt,
        f"{label}: the model was not given the ChatML prompt of the messages",
        detail + f"expected prompt: {prompt!r}\n     got prompt: {call.prompt!r}\n",
    )
    check.that(
        call.max_tokens == limit_of(body),
        f"{label}: the model was called with the wrong max_tokens",
        detail + f"expected: {limit_of(body)!r}\n     got: {call.max_tokens!r}\n",
    )
    check.that(
        len(call.pieces) == want.count,
        f"{label}: the server read {len(call.pieces)} rows of logits from the model, "
        "which is not the number this generation needs",
        detail + f"expected {want.count} rows: one per token chosen, end-of-sequence "
        "included, and none after it\n",
    )


def normalize_calls(parts, ids, label, detail):
    """Replace the id of every starting tool-call part by "call_#" after checking it."""
    out = []
    for part in parts:
        part = json.loads(json.dumps(part))
        if "id" in part:
            ident = part["id"]
            check.that(
                isinstance(ident, str) and ident.startswith("call_") and len(ident) > 5,
                f'{label}: a tool call\'s "id" must be a string that starts with "call_"',
                detail,
            )
            ids.append(ident)
            part["id"] = "call_#"
        out.append(part)
    return out


def complete(ctx, client, lm, body, rows, label):
    """A non-streaming request, checked against the statement's rules. Returns Expected."""
    want = expect(ctx, body, rows)
    index = len(lm.calls)
    response = send(client, "POST", URL, label, json=body)
    detail = describe(label, body) + (
        f"status: {response.status_code}\nresponse: {response.text[:800]}\n"
    )
    check.that(response.status_code == 200, f"{label}: the status is not 200", detail)
    check.that(
        response.headers.get("content-type", "").startswith("application/json"),
        f"{label}: the response is not application/json",
        detail,
    )
    try:
        obj = response.json()
    except ValueError:
        raise CheckFailure(f"{label}: the response body is not JSON", detail) from None
    check_base(obj, "chat.completion", body["model"], label, detail)
    check_model_call(lm, index, body, want, label, detail)
    choices = obj.get("choices")
    check.that(
        isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
        f'{label}: "choices" must be a list with one choice',
        detail,
    )
    choice = choices[0]
    message = choice.get("message")
    check.that(
        choice.get("index") == 0 and isinstance(message, dict),
        f"{label}: the choice needs index 0 and a message",
        detail,
    )
    detail += f"expected message: {json.dumps(want.message)}\n"
    check.that(
        message.get("role") == "assistant", f"{label}: the message's role is not assistant", detail
    )
    check.that(
        message.get("content") == want.message["content"],
        f"{label}: wrong message content",
        detail,
    )
    ids = []
    got_calls = message.get("tool_calls")
    if "tool_calls" in want.message:
        check.that(
            isinstance(got_calls, list) and all(isinstance(c, dict) for c in got_calls),
            f'{label}: the message needs a "tool_calls" list',
            detail,
        )
        check.that(
            normalize_calls(got_calls, ids, label, detail) == want.message["tool_calls"],
            f"{label}: wrong tool_calls in the message (type, function name, or arguments)",
            detail,
        )
        check.that(len(set(ids)) == len(ids), f"{label}: two tool calls share an id", detail)
    else:
        check.that(
            not got_calls,
            f'{label}: the message has "tool_calls" although no call was made',
            detail,
        )
    check.that(
        choice.get("finish_reason") == want.finish,
        f"{label}: wrong finish_reason",
        detail + f"expected finish_reason: {want.finish!r}\n",
    )
    check.that(
        obj.get("usage") == want.usage,
        f"{label}: wrong usage (prompt_tokens, completion_tokens, total_tokens)",
        detail + f"expected usage: {want.usage!r}\n",
    )
    return want


def check_chunks(events, lm, index, body, want, label):
    """The events of a finished stream against the expected deltas. Returns the chunks."""
    shown = "\n".join(f"  {e.data[:240]}" for e in events[:30]) or "  (none)"
    detail = describe(label, body) + f"events received (data):\n{shown}\n"
    check.that(
        bool(events) and events[-1].data == "[DONE]",
        f"{label}: the stream does not end with the event `data: [DONE]`",
        detail,
    )
    chunks = []
    for event in events[:-1]:
        try:
            chunks.append(json.loads(event.data))
        except ValueError:
            raise CheckFailure(f"{label}: an event's data is not JSON", detail) from None
    check_model_call(lm, index, body, want, label, detail)
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    for chunk in chunks:
        check_base(chunk, "chat.completion.chunk", body["model"], label, detail)
    check.that(
        len({chunk["id"] for chunk in chunks}) == 1
        and len({chunk["created"] for chunk in chunks}) == 1,
        f"{label}: all chunks of one stream must have the same id and the same created",
        detail,
    )
    with_choice = chunks[:-1] if include_usage else chunks
    for position, chunk in enumerate(with_choice):
        choices = chunk.get("choices")
        check.that(
            isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], dict)
            and choices[0].get("index") == 0
            and isinstance(choices[0].get("delta"), dict),
            f"{label}: every chunk but the usage chunk has one choice with index 0 and a delta",
            detail + f"chunk {position}: {json.dumps(chunk)[:300]}\n",
        )
        check.that(
            chunk.get("usage") is None,
            f'{label}: "usage" must be null (or absent) on every chunk but the usage chunk',
            detail + f"chunk {position}: {json.dumps(chunk)[:300]}\n",
        )
    check.that(
        len(with_choice) >= 2,
        f"{label}: a stream needs at least the role chunk and the finish chunk",
        detail,
    )
    deltas = [chunk["choices"][0]["delta"] for chunk in with_choice]
    reasons = [chunk["choices"][0].get("finish_reason") for chunk in with_choice]
    check.that(
        deltas[0].get("role") == "assistant"
        and not deltas[0].get("content")
        and not deltas[0].get("tool_calls"),
        f'{label}: the first chunk\'s delta must be {{"role": "assistant", "content": ""}}',
        detail,
    )
    check.that(
        all(delta.get("role") is None for delta in deltas[1:]),
        f"{label}: only the first chunk's delta carries the role",
        detail,
    )
    ids = []
    got = []
    for delta in deltas[1:-1]:
        seen = {}
        if delta.get("content") is not None:
            seen["content"] = delta["content"]
        if delta.get("tool_calls") is not None:
            parts = delta["tool_calls"]
            check.that(
                isinstance(parts, list) and all(isinstance(p, dict) for p in parts),
                f'{label}: "tool_calls" in a delta must be a list of objects',
                detail,
            )
            seen["tool_calls"] = normalize_calls(parts, ids, label, detail)
        got.append(seen)
    expected_text = "\n".join(f"  {json.dumps(delta)}" for delta in want.deltas[:30])
    detail += f"expected deltas between the role chunk and the finish chunk:\n{expected_text}\n"
    check.that(
        len(got) == len(want.deltas),
        f"{label}: got {len(got)} chunks between the role chunk and the finish chunk, "
        "which is not the expected number",
        detail,
    )
    for position, (have, need) in enumerate(zip(got, want.deltas, strict=True)):
        check.that(
            have == need,
            f"{label}: chunk {position + 1} does not carry the expected delta",
            detail + f"chunk {position + 1}: got {json.dumps(have)}, expected {json.dumps(need)}\n",
        )
    check.that(len(set(ids)) == len(ids), f"{label}: two tool calls share an id", detail)
    check.that(
        not deltas[-1].get("content") and not deltas[-1].get("tool_calls"),
        f"{label}: the finish chunk's delta must be empty",
        detail,
    )
    check.that(
        all(reason is None for reason in reasons[:-1]),
        f"{label}: finish_reason must be null on every chunk before the finish chunk",
        detail,
    )
    check.that(
        reasons[-1] == want.finish,
        f"{label}: wrong finish_reason on the finish chunk",
        detail + f"expected finish_reason: {want.finish!r}\n",
    )
    if include_usage:
        last = chunks[-1]
        check.that(
            last.get("choices") == [], f'{label}: the usage chunk must have "choices": []', detail
        )
        check.that(
            last.get("usage") == want.usage,
            f"{label}: wrong usage in the usage chunk",
            detail + f"expected usage: {want.usage!r}\n",
        )
    else:
        check.that(
            all(chunk.get("choices") != [] for chunk in chunks),
            f"{label}: a usage chunk was sent although include_usage was not set",
            detail,
        )
    return chunks


def stream(ctx, client, lm, body, rows, label):
    """A streaming request read to its end; checks every chunk. Returns Expected."""
    want = expect(ctx, body, rows)
    index = len(lm.calls)
    events = []
    try:
        with client.stream("POST", URL, json=body) as response:
            if response.status_code != 200:
                response.read()
                raise CheckFailure(
                    f"{label}: the status is not 200",
                    describe(label, body) + f"status: {response.status_code}\n"
                    f"response: {response.text[:400]}\n",
                )
            events.extend(iter_sse(response))
    except httpx.TimeoutException:
        raise CheckFailure(
            f"{label}: timed out reading the stream after {len(events)} events",
            "The response must end after `data: [DONE]`.\n",
        ) from None
    except httpx.HTTPError as exc:
        raise CheckFailure(f"{label}: the stream broke: {type(exc).__name__}", f"{exc}\n") from None
    check_chunks(events, lm, index, body, want, label)
    return want


def both(ctx, client, lm, body, rows, label):
    """The same request not streamed and streamed (with a usage chunk)."""
    lm.script = rows
    want = complete(ctx, client, lm, body, rows, label)
    streamed = {**body, "stream": True, "stream_options": {"include_usage": True}}
    stream(ctx, client, lm, streamed, rows, label + ", streamed")
    return want


def check_closed(lm, label):
    check.that(
        lm.wait_closed(timeout_s=2.0),
        f"{label}: the model's stream was left open after the response",
        "Call `await stream.aclose()` when the generation is over, also after "
        "end-of-sequence: the model does not know that the server stopped listening.\n",
    )


def next_event(events, what):
    try:
        return next(events)
    except StopIteration:
        raise CheckFailure(f"the stream ended before {what} arrived") from None
    except httpx.TimeoutException:
        raise CheckFailure(f"timed out waiting for {what}") from None
    except httpx.HTTPError as exc:
        raise CheckFailure(
            f"the connection broke while waiting for {what}: {type(exc).__name__}", f"{exc}\n"
        ) from None


def check_error(response, status, label, param="unchecked"):
    detail = (
        f"request: {label}\nstatus: {response.status_code}\n"
        f"content-type: {response.headers.get('content-type')!r}\nbody: {response.text[:300]!r}\n"
    )
    check.that(response.status_code == status, f"{label}: wrong status code", detail)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    check.that(
        isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"],
        f'{label}: the body must be {{"error": {{"message": "<text>", ...}}}}',
        detail,
    )
    if param == "unchecked":
        return
    check.that(
        error.get("type") == "invalid_request_error" and "param" in error and "code" in error,
        f'{label}: the error object needs "type": "invalid_request_error", "param", and "code"',
        detail,
    )
    check.that(
        error["param"] == param,
        f'{label}: "param" must name the offending field (null when the body itself is wrong)',
        detail + f"expected param: {param!r}\n",
    )


def check_invalid(client, lm, bodies):
    """Each body gets a 400 naming its field, and the model is never called."""
    calls = len(lm.calls)
    for body, param in bodies:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        label = f"POST {URL} with body {raw[:160]!r}"
        headers = {"content-type": "application/json"}
        response = send(client, "POST", URL, label, content=raw, headers=headers)
        check_error(response, 400, label, param)
    check.that(
        len(lm.calls) == calls,
        "the model was called for an invalid request",
        f"prompts: {[call.prompt for call in lm.calls[calls:]]!r}\n",
    )


# == Stage 1: chat completions ============================================================

HELLO = [*spell("Hello, world!"), ranked(EOS), ranked("?"), ranked("?")]


@case("the stage 1 example, not streamed: greedy tokens up to end-of-sequence", public=True,
      stage=S1)  # fmt: skip
def chat_example(ctx):
    # Rejects eos_not_counted and reads_past_eos.
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        want = complete(ctx, client, lm, body_of(), HELLO, "the example request")
        check_closed(lm, "the example request")
    # The statement's numbers, literally.
    check.equal(want.text, "Hello, world!", label="content in the statement")
    check.equal(
        want.usage,
        {"prompt_tokens": 8, "completion_tokens": 13, "total_tokens": 21},
        label="usage in the statement",
    )


@case("the stage 1 example, streamed with a limit of 4 tokens and a usage chunk", public=True,
      stage=S1)  # fmt: skip
def chat_example_stream(ctx):
    lm = ScriptedLM(HELLO)
    body = body_of(max_tokens=4, stream=True, stream_options={"include_usage": True})
    with serve(ctx, lm) as client:
        want = stream(ctx, client, lm, body, HELLO, "the example request, streamed")
        check_closed(lm, "the example request, streamed")
    check.equal(want.text, "Hell", label="content in the statement")
    check.equal(want.finish, "length", label="finish_reason in the statement")


@case("finish_reason is stop exactly when end-of-sequence was chosen", public=True, stage=S1)
def finish_reasons(ctx):
    # Rejects finish_from_count: with the limit at 13, the 13th token is end-of-sequence.
    lm = ScriptedLM(HELLO)
    table = [
        ({"max_tokens": 12}, "length"),  # the twelve tokens of the text, no room for EOS
        ({"max_tokens": 13}, "stop"),  # the limit is reached by choosing EOS
        ({"max_completion_tokens": 14}, "stop"),
        ({"max_completion_tokens": 3, "max_tokens": 50}, "length"),  # the new name wins
        ({"max_completion_tokens": None, "max_tokens": 5}, "length"),  # null is absent
        ({}, "stop"),
    ]
    with serve(ctx, lm) as client:
        for fields, reason in table:
            for streamed in (False, True):
                body = body_of(**fields, stream=streamed)
                label = f"{'streamed' if streamed else 'not streamed'}, {fields!r}"
                run = stream if streamed else complete
                want = run(ctx, client, lm, body, HELLO, label)
                assert want.finish == reason, (fields, want.finish)
        # A model whose rows run out without end-of-sequence was cut off.
        lm.script = HELLO[:5]
        want = complete(ctx, client, lm, body_of(), HELLO[:5], "rows end without EOS")
        assert want.finish == "length"
        check_closed(lm, "after the last request")


@case("the highest logit wins, and the lowest id among equals", public=True, stage=S1)
def argmax_and_ties(ctx):
    # Rejects ties_highest_id.
    a, b, c = sorted(token(piece) for piece in "abc")
    tie = base_row()
    tie[a] = tie[b] = tie[c] = 3.0
    late = base_row()
    late[c] = 2.5
    eos_tie = base_row()
    eos_tie[VOCAB.eos_id] = eos_tie[c] = 1.0  # end-of-sequence has id 1: it wins the tie
    rows = [tie, late, tie, eos_tie, tie]
    lm = ScriptedLM(rows)
    with serve(ctx, lm) as client:
        want = both(ctx, client, lm, body_of(), rows, "rows with tied logits")
    assert want.finish == "stop" and want.count == 4


INVALID_CHAT = [
    (b"{not json", None),
    (b"", None),
    (b"[]", None),
    ({"model": "m"}, "messages"),
    ({"model": "m", "messages": []}, "messages"),
    ({"model": "m", "messages": [{"role": "user"}]}, "messages"),
    ({"model": "m", "messages": [{"role": "wizard", "content": "hi"}]}, "messages"),
    ({"messages": CONVERSATION}, "model"),
    ({"model": "", "messages": CONVERSATION}, "model"),
    ({"model": "m", "messages": CONVERSATION, "max_tokens": 0}, "max_tokens"),
    ({"model": "m", "messages": CONVERSATION, "max_tokens": True}, "max_tokens"),
    ({"model": "m", "messages": CONVERSATION, "max_completion_tokens": 2.5}, "max_completion_tokens"),
    ({"model": "m", "messages": CONVERSATION, "stream": "yes"}, "stream"),
    ({"model": "m", "messages": CONVERSATION, "stream": True, "stream_options": []}, "stream_options"),
]  # fmt: skip


@case("routing and validation: 404, 405, and 400 with an OpenAI-shaped error", public=True,
      stage=S1)  # fmt: skip
def routing_and_validation(ctx):
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        for method, path in [("POST", "/v1/completions"), ("GET", "/"), ("POST", URL + "/x")]:
            response = send(client, method, path, f"{method} {path}", json=body_of())
            check_error(response, 404, f"{method} {path}")
        for method in ["GET", "PUT"]:
            response = send(client, method, URL, f"{method} {URL}", json=body_of())
            check_error(response, 405, f"{method} {URL}")
            allow = response.headers.get("allow", "")
            check.that(
                "POST" in [m.strip() for m in allow.split(",")],
                f"{method} {URL}: the 405 response has no `allow: POST` header",
                f"allow header: {allow!r}\n",
            )
        check_invalid(client, lm, INVALID_CHAT)
        # Still healthy; unknown fields and nulls are ignored.
        body = body_of(temperature=0.2, user="u-1", stream=None, stream_options=None)
        complete(ctx, client, lm, body, HELLO, "a valid request after the invalid ones")


@case("each token is flushed before the next row of logits is asked for", stage=S1)
def flushes_each_token(ctx):
    # Rejects buffers_stream. The gate lets one row through at a time.
    rows = [*spell("ab cd"), ranked(EOS)]
    lm = ScriptedLM(rows, gated=True, gate_timeout_s=1.5)
    body = body_of(stream=True, stream_options={"include_usage": True})
    want = expect(ctx, body, rows)
    with serve(ctx, lm) as client, client.stream("POST", URL, json=body) as response:
        events = iter_sse(response)
        received = []
        for i, text in enumerate(want.deltas):
            lm.release(1)
            while True:
                event = next_event(events, f"the chunk for token {i}")
                check.that(
                    not lm.stalled,
                    "chunks are not flushed as tokens are chosen: a chunk reached the client "
                    "only after the model had given up waiting",
                    "Send every chunk in its own http.response.body message with "
                    "more_body=True, as soon as its token is chosen.\n",
                )
                received.append(event)
                choices = event.json().get("choices") or [{}]
                if (choices[0].get("delta") or {}).get("content"):
                    break
            check.equal(choices[0]["delta"]["content"], text["content"], label=f"chunk {i}")
            check.equal(len(lm.calls[0].pieces), i + 1, label="rows read when the chunk arrived")
        lm.release()
        received.extend(events)
    check_chunks(received, lm, 0, body, want, "a stream released one row at a time")


@case("seeded rows of logits, streamed and not, equal the greedy replay", stage=S1)
def chat_random(ctx):
    lm = ScriptedLM(HELLO)
    favorites = ["e", "t", " ", "a", "o", ", ", "n", "true", '"', "{", "!"]
    with serve(ctx, lm) as client:
        for trial in range(6):
            rng = ctx.child_rng(trial)
            rows = random_rows(rng, 40, favorites, eos_from=int(rng.integers(8, 36)))
            fields = {} if trial % 2 else {"max_tokens": int(rng.integers(4, 30))}
            messages = [{"role": "user", "content": f"trial   {trial}\nplease  answer"}]
            both(ctx, client, lm, body_of(messages, **fields), rows, f"trial {trial}")
        check_closed(lm, "after the last request")


# == Stage 2: schema-constrained output ===================================================

FLAGS = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "n": {"type": "integer"}},
    "required": ["ok"],
}
# What the model of the statement's example prefers, step by step.
FLAGS_ROWS = [
    ranked('{"'),
    ranked("o"),
    ranked("k"),
    ranked('": ', '":'),
    ranked('"', "true"),
    ranked(", ", ",", "}"),
    ranked('"'),
    ranked("n"),
    ranked('": ', '":'),
    ranked("0"),
    ranked("7", "}"),
    ranked("\n", EOS),
    ranked("\n", EOS),
]
TRIP = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "maxLength": 12},
        "days": {"type": "integer"},
        "unit": {"enum": ["C", "F"]},
        "stops": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "budget": {"type": "number"},
        "notes": {"type": "null"},
    },
    "required": ["city", "days"],
}
SCHEMAS = [
    FLAGS,
    TRIP,
    {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
    {"type": "string", "minLength": 2, "maxLength": 6},
    {"enum": ["red", "green", 3, True, None]},
    {"type": "number"},
    {
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "array", "items": {"enum": ["x", "y"]}},
            "c": {"type": "object", "properties": {"d": {"type": "integer"}}},
        },
    },
]
CLOSERS = ('"', ":", ",", "}", "]", EOS)


@case("the stage 2 example: the schema overrides what the model prefers", public=True, stage=S2)
def schema_example(ctx):
    # Rejects stream_unconstrained on its streamed half.
    lm = ScriptedLM(FLAGS_ROWS)
    body = body_of(response_format=schema_format(FLAGS))
    with serve(ctx, lm) as client:
        want = both(ctx, client, lm, body, FLAGS_ROWS, "the example request")
        check_closed(lm, "the example request")
        both(ctx, client, lm, {**body, "max_tokens": 5}, FLAGS_ROWS, "the example, max_tokens 5")
    check.equal(want.text, '{"ok":true,"n":0}', label="content in the statement")
    check.equal(want.finish, "stop", label="finish_reason in the statement")
    check.equal(want.count, 12, label="completion_tokens in the statement")
    assert expect(ctx, {**body, "max_tokens": 5}, FLAGS_ROWS).text == '{"ok":true'


@case("a model that writes prose and loose JSON still yields documents of the schema",
      public=True, stage=S2)  # fmt: skip
def prose_model(ctx):
    scripts = [
        (TRIP, 'Sure! {"city": "Oslo", "days": 3, "unit": "C"}'),
        (TRIP, '{"city":"Rome","days":"two","stops":["a","b"],"budget":1e3}'),
        (SCHEMAS[2], "[1, 2, 03]"),
        (SCHEMAS[3], "'hello world'"),
        (SCHEMAS[4], "blue"),
        (SCHEMAS[5], "+12.50"),
    ]
    lm = ScriptedLM(FLAGS_ROWS)
    with serve(ctx, lm) as client:
        for schema, text in scripts:
            rows = [*spell(text, *CLOSERS), *[ranked(EOS, *CLOSERS)] * 30]
            label = f"the model wants to write {text!r}"
            body = body_of(response_format=schema_format(schema))
            want = both(ctx, client, lm, body, rows, label)
            assert want.finish == "stop", (text, want.text)
            check_document(schema, want.text, label)


@case("a document cut off by the token limit is reported with finish_reason length",
      public=True, stage=S2)  # fmt: skip
def truncated_document(ctx):
    # Rejects constrained_finish_stop and eos_in_any_state.
    text = '{"city":"Oslo","days":365}'
    rows = [*spell(text), ranked(EOS), ranked(EOS)]
    n = len(VOCAB.encode(text))
    lm = ScriptedLM(rows)
    with serve(ctx, lm) as client:
        for limit, reason in [(3, "length"), (n - 1, "length"), (n, "length"), (n + 1, "stop")]:
            body = body_of(response_format=schema_format(TRIP), max_tokens=limit)
            want = both(ctx, client, lm, body, rows, f"max_tokens {limit}")
            assert want.finish == reason, (limit, want.finish)
        # A model that wants to stop at every step stops at the first complete document.
        eager = [ranked(EOS, '"', "}", "]", "7", "a")] * 40
        for schema in (SCHEMAS[2], SCHEMAS[3], FLAGS, SCHEMAS[5]):
            body = body_of(response_format=schema_format(schema))
            label = f"a model that prefers end-of-sequence, schema {json.dumps(schema)[:60]}"
            want = both(ctx, client, lm, body, eager, label)
            assert want.finish == "stop", (schema, want.text)
            check_document(schema, want.text, label)


INVALID_FORMAT = [
    "json",
    {"type": "json_object"},
    {"type": "json_schema"},
    {"type": "json_schema", "json_schema": {"schema": FLAGS}},
    {"type": "json_schema", "json_schema": {"name": "", "schema": FLAGS}},
    {"type": "json_schema", "json_schema": {"name": "answer"}},
    {"type": "json_schema", "json_schema": {"name": "answer", "schema": "object"}},
    schema_format({"type": "date"}),
    schema_format({"properties": {"a": {"type": "integer"}}}),
    schema_format({"type": "array", "items": {"type": ["integer", "null"]}}),
    schema_format({"type": "object", "properties": {"a": {"type": "tuple"}}}),
]


@case("response_format: text and null leave the model free; a bad format is a 400", public=True,
      stage=S2)  # fmt: skip
def response_format_validation(ctx):
    # Rejects unsupported_schema_500.
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        for spec in ({"type": "text"}, None):
            want = both(ctx, client, lm, body_of(response_format=spec), HELLO, f"format {spec!r}")
            assert want.text == "Hello, world!"
        check_invalid(
            client,
            lm,
            [(body_of(response_format=spec), "response_format") for spec in INVALID_FORMAT],
        )
        body = body_of(response_format=schema_format(FLAGS))
        both(ctx, client, lm, body, FLAGS_ROWS, "a valid schema after the invalid ones")


@case("tokens that span structure: quotes with braces, colons, and split keywords", stage=S2)
def spanning_tokens(ctx):
    # Tokens such as `"}`, `":`, `{"`, `tru`, and `nul` cross the boundaries of the grammar.
    lm = ScriptedLM(HELLO)
    schema = {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "t": {"type": "boolean"},
            "u": {"type": "null"},
            "v": {"type": "string"},
        },
        "required": ["s"],
    }
    scripts = [
        ['{"', "s", '":', '"', "a", '"}', EOS],
        ['{"', "s", '": ', '":', '"', "{", '"}', '":', '"', ", ", '"', "t", '":', "tru", "e", "}"],
        ['{"', "s", '":', '"', '"', ",", '"', "u", '":', "nu", "l", "nul", "l", '"}', "}", EOS],
        ['{"', "s", '":', '"', "\\", '"}', '"', '"}', ",", '"', "v", '":', '"', "é", '"}', EOS],
    ]
    with serve(ctx, lm) as client:
        for number, pieces in enumerate(scripts):
            rows = [ranked(piece, *CLOSERS) for piece in pieces] + [ranked(EOS, *CLOSERS)] * 20
            label = f"script {number}"
            body = body_of(response_format=schema_format(schema))
            want = both(ctx, client, lm, body, rows, label)
            if want.finish == "stop":
                check_document(schema, want.text, label)


@case("seeded logits on seven schemas, with and without a token limit", stage=S2)
def schema_random(ctx):
    lm = ScriptedLM(HELLO)
    favorites = ['"', '"}', '":', '": ', ",", ", ", "}", "]", "true", "nul", "1", "0", "e", "a"]
    stopped = 0
    with serve(ctx, lm) as client:
        for trial in range(14):
            rng = ctx.child_rng(trial)
            schema = SCHEMAS[trial % len(SCHEMAS)]
            rows = random_rows(rng, 60, favorites)
            for i, row in enumerate(rows):
                # The longer the generation, the more the model wants to close and stop.
                for piece in ('"', "}", "]"):
                    row[token(piece)] += 0.4 * i
                row[VOCAB.eos_id] = -3.0 + 0.5 * i
            fields = {"max_tokens": int(rng.integers(3, 40))} if trial >= 7 else {}
            label = f"trial {trial}"
            body = body_of(response_format=schema_format(schema), **fields)
            want = both(ctx, client, lm, body, rows, label)
            if want.finish == "stop":
                stopped += 1
                check_document(schema, want.text, label)
    assert stopped >= 8, stopped


# == Stage 3: tool calls ==================================================================

WEATHER = tool(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"enum": ["C", "F"]}},
        "required": ["city"],
    },
    "The weather in a city.",
)
CLOCK = tool("get_time")
CALL_TEXT = 'Checking.<tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>'
CALL_ROWS = [*spell(CALL_TEXT), ranked(EOS), ranked("?")]


@case("the stage 3 example, not streamed: text, then one tool call", public=True, stage=S3)
def tool_example(ctx):
    # Rejects tool_finish_stop.
    lm = ScriptedLM(CALL_ROWS)
    body = body_of(tools=[WEATHER, CLOCK])
    with serve(ctx, lm) as client:
        want = complete(ctx, client, lm, body, CALL_ROWS, "the example request")
        check_closed(lm, "the example request")
    check.equal(want.message["content"], "Checking.", label="content in the statement")
    check.equal(
        want.message["tool_calls"][0]["function"],
        {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
        label="function in the statement",
    )
    check.equal(want.finish, "tool_calls", label="finish_reason in the statement")


@case("the stage 3 example, streamed: one start delta, then argument fragments", public=True,
      stage=S3)  # fmt: skip
def tool_example_stream(ctx):
    # Rejects id_in_every_delta.
    lm = ScriptedLM(CALL_ROWS)
    body = body_of(tools=[WEATHER, CLOCK], stream=True, stream_options={"include_usage": True})
    with serve(ctx, lm) as client:
        want = stream(ctx, client, lm, body, CALL_ROWS, "the example request, streamed")
    starts = [d for d in want.deltas if "id" in d.get("tool_calls", [{}])[0]]
    check.equal(len(starts), 1, label="start deltas in the statement")


@case("a named tool_choice constrains the arguments to the tool's parameters", public=True,
      stage=S3)  # fmt: skip
def named_tool(ctx):
    # Rejects named_choice_unconstrained.
    lm = ScriptedLM(CALL_ROWS)
    text = 'I will look it up: {"city": "Oslo", "unit": "K"}'
    rows = [*spell(text, *CLOSERS), *[ranked(EOS, *CLOSERS)] * 20]
    schema = WEATHER["function"]["parameters"]
    with serve(ctx, lm) as client:
        body = body_of(tools=[WEATHER, CLOCK], tool_choice=named("get_weather"))
        want = both(ctx, client, lm, body, rows, "tool_choice names get_weather")
        assert want.finish == "tool_calls", want.finish
        check.that(want.message["content"] is None, "internal: content must be null")
        check_document(schema, want.message["tool_calls"][0]["function"]["arguments"], "arguments")
        # A tool without parameters takes the empty object.
        body = body_of(tools=[WEATHER, CLOCK], tool_choice=named("get_time"))
        want = both(ctx, client, lm, body, rows, "tool_choice names get_time")
        assert want.message["tool_calls"][0]["function"]["arguments"] == "{}"
        # Cut off inside the arguments.
        body = body_of(tools=[WEATHER], tool_choice=named("get_weather"), max_tokens=5)
        want = both(ctx, client, lm, body, rows, "named tool, max_tokens 5")
        assert want.finish == "length", want.finish


INVALID_TOOLS = [
    ({"tools": "get_weather"}, "tools"),
    ({"tools": []}, "tools"),
    ({"tools": [{"type": "function"}]}, "tools"),
    ({"tools": [{"type": "retrieval", "function": {"name": "f"}}]}, "tools"),
    ({"tools": [{"type": "function", "function": {"name": ""}}]}, "tools"),
    ({"tools": [{"type": "function", "function": {"name": "f", "parameters": []}}]}, "tools"),
    ({"tools": [WEATHER], "tool_choice": "always"}, "tool_choice"),
    ({"tools": [WEATHER], "tool_choice": "required"}, "tool_choice"),
    ({"tools": [WEATHER], "tool_choice": named("get_time")}, "tool_choice"),
    ({"tools": [WEATHER], "tool_choice": {"type": "function"}}, "tool_choice"),
    ({"tool_choice": "auto"}, "tool_choice"),
    ({"tool_choice": named("get_weather")}, "tool_choice"),
    ({"tools": [tool("f", {"type": "date"})], "tool_choice": named("f")}, "tools"),
]


@case("without tools, or with tool_choice none, the markup is ordinary text", public=True,
      stage=S3)  # fmt: skip
def tools_off_and_validation(ctx):
    # Rejects parses_without_tools.
    lm = ScriptedLM(CALL_ROWS)
    with serve(ctx, lm) as client:
        for fields in ({}, {"tools": [WEATHER], "tool_choice": "none"}, {"tool_choice": "none"}):
            want = both(ctx, client, lm, body_of(**fields), CALL_ROWS, f"fields {fields!r}")
            assert want.text == CALL_TEXT and want.finish == "stop"
        # Tools are offered and the model answers in plain text.
        plain = [*spell("It is <b>sunny</b> at 5 < 6."), ranked(EOS)]
        want = both(ctx, client, lm, body_of(tools=[WEATHER]), plain, "tools, no call")
        assert want.finish == "stop" and "tool_calls" not in want.message
        # An unsupported parameters schema matters only when the tool is chosen by name.
        odd = [tool("f", {"type": "date"}), WEATHER]
        both(ctx, client, lm, body_of(tools=odd), CALL_ROWS, "unused odd parameters")
        check_invalid(client, lm, [(body_of(**fields), param) for fields, param in INVALID_TOOLS])


@case("several calls, held-back text, and strings that contain braces and tags", stage=S3)
def tool_streams(ctx):
    texts = [
        (
            'One <tool_call>{"name":"a","arguments":{}}</tool_call> and two '
            '<tool_call> { "name" : "b\\"c" , "arguments" : {"x": [1, {"y": "}"}]} } </tool_call><'
        ),
        (
            '<<tool_call>{"name": "get_weather", "arguments": {"note": "a \\"}\\" </tool_call>"}}'
            "</tool_call></tool_call> done <tool"
        ),
        (
            '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "get_time", "arguments": {"tz": "UTC"}}</tool_call>'
        ),
        "No call here, only a < and a <tool_cal",
    ]
    lm = ScriptedLM(CALL_ROWS)
    with serve(ctx, lm) as client:
        for number, text in enumerate(texts):
            rows = [*spell(text), ranked(EOS)]
            body = body_of(tools=[WEATHER, CLOCK], tool_choice="auto")
            want = both(ctx, client, lm, body, rows, f"text {number}")
            assert want.finish == ("stop" if number == 3 else "tool_calls"), (number, want.finish)
        # Cut off by the limit: inside the name (no call yet), then inside the arguments.
        rows = [*spell(CALL_TEXT), ranked(EOS)]
        for number, (limit, calls) in enumerate([(14, 0), (22, 0), (40, 1), (len(rows) - 1, 1)]):
            body = body_of(tools=[WEATHER], max_completion_tokens=limit)
            want = both(ctx, client, lm, body, rows, f"limit {number}")
            assert want.finish == "length", (limit, want.finish)
            assert len(want.message.get("tool_calls", [])) == calls, (limit, want.message)


@case("named tools on seeded logits: every finished call has conforming arguments", stage=S3)
def named_random(ctx):
    tools = [
        tool("plan_trip", TRIP),
        tool("flags", FLAGS),
        tool("pick", {"enum": ["red", "green", 3, True, None]}),
        tool("noop"),
    ]
    favorites = ['"', '"}', '":', ",", "}", "]", "true", "1", "0", "e", "a", "<", "{"]
    lm = ScriptedLM(CALL_ROWS)
    with serve(ctx, lm) as client:
        for trial in range(8):
            rng = ctx.child_rng(trial)
            chosen = tools[trial % len(tools)]["function"]
            rows = random_rows(rng, 50, favorites)
            for i, row in enumerate(rows):
                for piece in ('"', "}", "]"):
                    row[token(piece)] += 0.4 * i
                row[VOCAB.eos_id] = -3.0 + 0.5 * i
            fields = {"max_tokens": int(rng.integers(2, 30))} if trial % 3 == 2 else {}
            label = f"trial {trial}"
            body = body_of(tools=tools, tool_choice=named(chosen["name"]), **fields)
            want = both(ctx, client, lm, body, rows, label)
            if want.finish == "tool_calls":
                schema = chosen.get("parameters", {"type": "object", "properties": {}})
                arguments = want.message["tool_calls"][0]["function"]["arguments"]
                check_document(schema, arguments, label)


# == Stage 4: cancellation and the compilation cache =======================================


class CompileSpy:
    """Records every automaton that `CharDFA.from_regex` builds while the case runs."""

    def __init__(self, ctx):
        self.dfas = []
        original = CharDFA.__dict__["from_regex"]

        def from_regex(cls, pattern):
            dfa = original.__func__(cls, pattern)
            self.dfas.append(dfa)
            return dfa

        CharDFA.from_regex = classmethod(from_regex)
        ctx.defer(lambda: setattr(CharDFA, "from_regex", original))

    def read(self):
        return len(self.dfas), sum(dfa.counters.snapshot()["steps"] for dfa in self.dfas)


def counted(spy, run, label, *, compilations):
    """Run one request and check how many compilations and DFA steps it caused."""
    before = spy.read()
    result = run()
    after = spy.read()
    compiled, steps = after[0] - before[0], after[1] - before[1]
    detail = (
        f"request: {label}\nCharDFA.from_regex calls during the request: {compiled}\n"
        f"dfa.step calls during the request, over all compiled automata: {steps}\n"
    )
    if compilations == 0:
        check.cost({"compilations": compiled}, "compilations", exactly=0)
        check.that(
            steps == 0,
            f"{label}: the schema was compiled before, but this request stepped its "
            f"automaton {steps} times",
            detail + "Build the whole token index when the schema is compiled, and keep it "
            "with the automaton: a later request only looks states up.\n",
        )
    else:
        check.that(
            compiled == compilations,
            f"{label}: a schema seen for the first time was compiled {compiled} times "
            "with CharDFA.from_regex",
            detail + f"expected exactly {compilations}\n",
        )
    return result


def unique_schema(ctx, tag):
    """A schema no other case (and no other run of this case's app) has compiled."""
    return {
        "type": "object",
        "properties": {
            f"k{ctx.seed % 100000}{tag}": {"type": "integer"},
            "ok": {"type": "boolean"},
            "s": {"type": "string", "maxLength": 8},
        },
        "required": ["ok"],
    }


def schema_rows(rng):
    favorites = ['"', '"}', '":', ",", "}", "true", "1", "0", "a", "s", "o", "k"]
    rows = random_rows(rng, 50, favorites)
    for i, row in enumerate(rows):
        for piece in ('"', "}"):
            row[token(piece)] += 0.4 * i
        row[VOCAB.eos_id] = -3.0 + 0.5 * i
    return rows


@case("the second request with a schema compiles nothing and steps no automaton", public=True,
      stage=S4)  # fmt: skip
def compiles_once(ctx):
    # Rejects cache_by_identity (and a server without a cache).
    spy = CompileSpy(ctx)
    schema = unique_schema(ctx, "a")
    rows = schema_rows(ctx.child_rng(0))
    lm = ScriptedLM(rows)
    body = body_of(response_format=schema_format(schema))
    streamed = {**body, "stream": True}
    with serve(ctx, lm) as client:
        counted(spy, lambda: complete(ctx, client, lm, body, rows, "first"),
                "the first request with the schema", compilations=1)  # fmt: skip
        counted(spy, lambda: complete(ctx, client, lm, body, rows, "second"),
                "the second request with the schema", compilations=0)  # fmt: skip
        counted(spy, lambda: stream(ctx, client, lm, streamed, rows, "third, streamed"),
                "the third request with the schema, streamed", compilations=0)  # fmt: skip


@case("a client that disconnects mid-stream stops the generation", public=True, stage=S4)
def client_disconnect(ctx):
    # Rejects ignores_disconnect. The model would take about a second to run dry.
    rows = [ranked("w")] * 200
    lm = ScriptedLM(rows, delay_s=0.005)
    with serve(ctx, lm) as client:
        with client.stream("POST", URL, json=body_of(stream=True)) as response:
            events = iter_sse(response)
            for i in range(4):
                next_event(events, f"chunk {i}")
        check_stopped(lm, len(rows))
        lm.script, lm.delay_s = HELLO, 0.0
        complete(ctx, client, lm, body_of(), HELLO, "a request after a disconnect")


def check_stopped(lm, total):
    closed = lm.wait_closed(timeout_s=5.0)
    check.that(
        closed,
        "after the client disconnected, the model's stream was left open",
        "Close it with `await stream.aclose()` when you stop streaming.\n",
    )
    check.that(
        len(lm.calls[-1].pieces) < total,
        "the server kept generating after the client had disconnected",
        f"The client read 4 chunks and closed the connection; the server went on to pull all "
        f"{total} rows from the model.\nWatch receive() for the http.disconnect message "
        "while streaming.\n",
    )


@case("the cache is keyed by the schema's text: order of properties, spelling, and reuse",
      stage=S4)  # fmt: skip
def cache_key(ctx):
    # Rejects cache_sorted_keys: two schemas whose properties are declared in a different
    # order are different languages.
    spy = CompileSpy(ctx)
    tag = ctx.seed % 100000
    first = {
        "type": "object",
        "properties": {f"b{tag}": {"type": "boolean"}, f"a{tag}": {"type": "boolean"}},
        "required": [f"b{tag}", f"a{tag}"],
    }
    swapped = {
        "type": "object",
        "properties": {f"a{tag}": {"type": "boolean"}, f"b{tag}": {"type": "boolean"}},
        "required": [f"b{tag}", f"a{tag}"],
    }
    rows = [ranked(EOS, "true", '{"', '"', ":", ",", "}", "a", "b", *str(tag))] * 40
    lm = ScriptedLM(rows)
    with serve(ctx, lm) as client:

        def ask(schema, label, **fields):
            body = body_of(response_format=schema_format(schema), **fields)
            want = complete(ctx, client, lm, body, rows, label)
            check_document(schema, want.text, label)
            return want

        one = counted(spy, lambda: ask(first, "schema 1"), "schema 1", compilations=1)
        two = counted(spy, lambda: ask(swapped, "schema 2"), "schema 2", compilations=1)
        assert one.text != two.text
        counted(spy, lambda: ask(first, "schema 1, again"), "schema 1, again", compilations=0)
        counted(spy, lambda: ask(swapped, "schema 2, again"), "schema 2, again", compilations=0)

        # The same schema as the parameters of a named tool, in a body spelled with
        # other whitespace and under another format name: still the same schema.
        def as_tool():
            body = body_of(tools=[tool("f", first)], tool_choice=named("f"))
            raw = json.dumps(body, indent=3).encode()
            index = len(lm.calls)
            response = send(client, "POST", URL, "as tool parameters", content=raw,
                            headers={"content-type": "application/json"})  # fmt: skip
            check.equal(response.status_code, 200, label="status of the tool request")
            arguments = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
            check.equal(arguments["arguments"], one.text, label="arguments of the tool call")
            check.equal(len(lm.calls), index + 1, label="model calls")

        counted(spy, as_tool, "schema 1 in another request", compilations=0)
        other = schema_format(first, name="another_name")
        counted(
            spy,
            lambda: complete(ctx, client, lm, body_of(response_format=other), rows, "renamed"),
            "schema 1 in a third request",
            compilations=0,
        )


@case("a compiled schema serves any later generation without stepping its automaton", stage=S4)
def index_is_complete(ctx):
    # Rejects index_per_request, and an index that is filled in state by state as
    # generations reach new states.
    spy = CompileSpy(ctx)
    schema = unique_schema(ctx, "b")
    lm = ScriptedLM(HELLO)
    body = body_of(response_format=schema_format(schema))
    with serve(ctx, lm) as client:
        short = [ranked(EOS, '{"', "o", "k", '":', "true", "}")] * 20
        lm.script = short
        counted(spy, lambda: complete(ctx, client, lm, body, short, "first"),
                "the first request with the schema", compilations=1)  # fmt: skip
        for trial in range(4):
            rows = schema_rows(ctx.child_rng(trial))
            lm.script = rows
            run = stream if trial % 2 else complete
            fields = {"stream": True} if trial % 2 else {}
            label = f"another generation with the schema ({trial})"
            want = counted(
                spy,
                lambda run=run, fields=fields, rows=rows, label=label: run(
                    ctx, client, lm, {**body, **fields}, rows, label
                ),
                label,
                compilations=0,
            )
            if want.finish == "stop":
                check_document(schema, want.text, label)


@case("disconnecting from a constrained stream stops the model and keeps the cache", stage=S4)
def disconnect_constrained(ctx):
    spy = CompileSpy(ctx)
    schema = {"type": "array", "items": {"enum": [f"v{ctx.seed % 100000}"]}}
    rows = [ranked(",", "[", '"', "v", *str(ctx.seed % 100000))] * 300
    lm = ScriptedLM(rows, delay_s=0.005)
    body = body_of(response_format=schema_format(schema), stream=True)
    with serve(ctx, lm) as client:
        with client.stream("POST", URL, json=body) as response:
            events = iter_sse(response)
            for i in range(5):
                next_event(events, f"chunk {i}")
        check_stopped(lm, len(rows))
        lm.delay_s = 0.0
        limited = {**body, "max_tokens": 30}
        counted(spy, lambda: stream(ctx, client, lm, limited, rows, "after the disconnect"),
                "the same schema after a disconnect", compilations=0)  # fmt: skip
        # The same language as the parameters of a named tool: its stream stops too.
        lm.delay_s = 0.005
        named_body = body_of(tools=[tool("g", schema)], tool_choice=named("g"), stream=True)
        with client.stream("POST", URL, json=named_body) as response:
            events = iter_sse(response)
            for i in range(5):
                next_event(events, f"chunk {i} of the tool stream")
        check_stopped(lm, len(rows))
