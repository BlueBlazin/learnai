"""Tests for openai-chat-endpoint.

Every case builds the app with `make_app(lm)`, where `lm` is the kit's ScriptedLM,
serves it on a local socket, and talks to it over HTTP. The model's record of how it
was used (`lm.calls`: the prompt and max_tokens of each call, the pieces produced, the
finish reason, whether the stream was closed) is what the responses are compared with.
Incremental delivery is checked with the model's gate, by order and never by time.
"""

import json

import httpx

from whetstone.judge import CheckFailure, case, check, iter_sse
from whetstone.kit import ScriptedLM

URL = "/v1/chat/completions"
HELLO = ["Hello", ",", " world", "!"]
CONVERSATION = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Say hello"},
]


def render_prompt(messages):
    """The statement's chat template (ChatML)."""
    turns = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(turns) + "<|im_start|>assistant\n"


def body_of(messages=CONVERSATION, model="tiny-chat", **fields):
    return {"model": model, "messages": messages, **fields}


def serve(ctx, lm):
    return ctx.serve(ctx.solution.make_app(lm))


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
    return f"request: {label}\nbody: {json.dumps(body)[:400]}\n"


def check_base(obj, object_type, model, label, detail):
    """id, object, created, model: the fields every response object carries."""
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


def usage_of(lm, call):
    prompt_tokens = lm.count_tokens(call.prompt)
    completion_tokens = len(call.pieces)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def check_call(lm, index, body, label, detail):
    """The model must have been called once more, with the rendered prompt and the limit."""
    check.that(
        len(lm.calls) == index + 1,
        f"{label}: the model must be called exactly once per valid request",
        detail + f"lm.stream was called {len(lm.calls)} times in total, expected {index + 1}\n",
    )
    call = lm.calls[index]
    want_prompt = render_prompt(body["messages"])
    check.that(
        call.prompt == want_prompt,
        f"{label}: the model was not given the ChatML prompt of the messages",
        detail + f"expected prompt: {want_prompt!r}\n     got prompt: {call.prompt!r}\n",
    )
    limit = body.get("max_completion_tokens")
    if limit is None:
        limit = body.get("max_tokens")
    check.that(
        call.max_tokens == limit,
        f"{label}: the model was called with the wrong max_tokens",
        detail + f"expected max_tokens: {limit!r}\n     got max_tokens: {call.max_tokens!r}\n",
    )
    return call


def complete(client, lm, body, label):
    """A non-streaming request; checks the whole response against the model's record."""
    index = len(lm.calls)
    response = send(client, "POST", URL, label, json=body)
    detail = describe(label, body) + (
        f"status: {response.status_code}\nresponse: {response.text[:600]}\n"
    )
    check.that(response.status_code == 200, f"{label}: expected status 200", detail)
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
    call = check_call(lm, index, body, label, detail)
    detail += f"pieces the model produced: {call.pieces!r}\n"
    choices = obj.get("choices")
    check.that(
        isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
        f'{label}: "choices" must be a list with one choice',
        detail,
    )
    choice = choices[0]
    check.that(choice.get("index") == 0, f"{label}: the choice's index must be 0", detail)
    message = choice.get("message")
    check.that(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("content") == "".join(call.pieces),
        f"{label}: choices[0].message must have the role assistant and the generated text",
        detail,
    )
    check.that(
        choice.get("finish_reason") == call.finish_reason,
        f"{label}: wrong finish_reason",
        detail + f"expected finish_reason: {call.finish_reason!r}\n",
    )
    check.that(
        obj.get("usage") == usage_of(lm, call),
        f"{label}: wrong usage (prompt_tokens, completion_tokens, total_tokens)",
        detail + f"expected usage: {usage_of(lm, call)!r}\n",
    )
    return obj


def check_chunks(events, lm, index, body, label):
    """The events of a finished stream, against the model's record. Returns the chunks."""
    shown = "\n".join(f"  {e.data[:200]}" for e in events[:14]) or "  (none)"
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
    call = check_call(lm, index, body, label, detail)
    detail += f"pieces the model produced: {call.pieces!r}\n"
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    expected = 2 + len(call.pieces) + (1 if include_usage else 0)
    check.that(
        len(chunks) == expected,
        f"{label}: wrong number of chunks (a role chunk, one chunk per piece, a finish chunk"
        + (", a usage chunk)" if include_usage else "; no usage chunk was asked for)"),
        detail + f"expected {expected} chunks before [DONE], got {len(chunks)}\n",
    )
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
    deltas = [chunk["choices"][0]["delta"] for chunk in with_choice]
    reasons = [chunk["choices"][0].get("finish_reason") for chunk in with_choice]
    check.that(
        deltas[0].get("role") == "assistant" and not deltas[0].get("content"),
        f'{label}: the first chunk\'s delta must be {{"role": "assistant", "content": ""}}',
        detail,
    )
    check.that(
        all(delta.get("role") is None for delta in deltas[1:]),
        f"{label}: only the first chunk's delta carries the role",
        detail,
    )
    check.that(
        [delta.get("content") for delta in deltas[1:-1]] == call.pieces,
        f"{label}: the content deltas must be the generated pieces, one per chunk, in order",
        detail,
    )
    check.that(
        not deltas[-1].get("content"),
        f"{label}: the finish chunk's delta must be empty",
        detail,
    )
    check.that(
        all(reason is None for reason in reasons[:-1]),
        f"{label}: finish_reason must be null on every chunk before the finish chunk",
        detail,
    )
    check.that(
        reasons[-1] == call.finish_reason,
        f"{label}: wrong finish_reason on the finish chunk",
        detail + f"expected finish_reason: {call.finish_reason!r}\n",
    )
    if include_usage:
        last = chunks[-1]
        check.that(
            last.get("choices") == [],
            f'{label}: the usage chunk must have "choices": []',
            detail,
        )
        check.that(
            last.get("usage") == usage_of(lm, call),
            f"{label}: wrong usage in the usage chunk",
            detail + f"expected usage: {usage_of(lm, call)!r}\n",
        )
    return chunks


def stream(client, lm, body, label):
    """A streaming request read to its end; checks every chunk."""
    index = len(lm.calls)
    events = []
    try:
        with client.stream("POST", URL, json=body) as response:
            events.extend(iter_sse(response))
    except httpx.TimeoutException:
        raise CheckFailure(
            f"{label}: timed out reading the stream after {len(events)} events",
            "The response must end after `data: [DONE]`.\n",
        ) from None
    except httpx.HTTPError as exc:
        raise CheckFailure(f"{label}: the stream broke: {type(exc).__name__}", f"{exc}\n") from None
    return check_chunks(events, lm, index, body, label)


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


@case("the worked example, not streamed: message, finish_reason, and usage", public=True)
def worked_example(ctx):
    # Rejects usage_in_characters.
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        obj = complete(client, lm, body_of(), "POST /v1/chat/completions")
    # The statement's numbers, literally.
    check.equal(obj["choices"][0]["message"]["content"], "Hello, world!", label="content")
    check.equal(obj["choices"][0]["finish_reason"], "stop", label="finish_reason")
    check.equal(
        obj["usage"],
        {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        label="usage",
    )


@case("the worked example, streamed with include_usage: chunks, usage chunk, [DONE]", public=True)
def worked_example_stream(ctx):
    # Rejects missing_done, role_in_every_chunk, new_id_per_chunk, usage_in_finish_chunk.
    lm = ScriptedLM(HELLO)
    body = body_of(max_tokens=2, stream=True, stream_options={"include_usage": True})
    with serve(ctx, lm) as client:
        chunks = stream(client, lm, body, "POST /v1/chat/completions (stream)")
    # The statement's numbers, literally: role, "Hello", ",", finish, usage.
    check.equal(len(chunks), 5, label="number of chunks before [DONE]")
    check.equal(chunks[3]["choices"][0]["finish_reason"], "length", label="finish_reason")
    check.equal(
        chunks[-1]["usage"],
        {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        label="usage",
    )


@case("finish_reason: length when the limit cut the generation, stop otherwise", public=True)
def finish_reasons(ctx):
    # Rejects finish_always_stop and ignores_max_completion_tokens.
    lm = ScriptedLM(HELLO)
    variants = [
        ({"max_tokens": 3}, "length"),
        ({"max_tokens": 4}, "length"),  # exactly the limit counts as cut off
        ({"max_tokens": 5}, "stop"),
        ({"max_completion_tokens": 2}, "length"),
        ({"max_completion_tokens": 9}, "stop"),
        ({"max_completion_tokens": 1, "max_tokens": 50}, "length"),  # the new name wins
        ({"max_completion_tokens": None, "max_tokens": 2}, "length"),  # null is absent
        ({"max_tokens": None}, "stop"),
    ]
    with serve(ctx, lm) as client:
        for fields, reason in variants:
            for streamed in (False, True):
                body = body_of(**fields, stream=streamed)
                label = f"{'streamed' if streamed else 'not streamed'}, {fields!r}"
                if streamed:
                    chunks = stream(client, lm, body, label)
                    got = chunks[-1]["choices"][0]["finish_reason"]
                else:
                    got = complete(client, lm, body, label)["choices"][0]["finish_reason"]
                # The model's own record agrees with the table above.
                assert lm.calls[-1].finish_reason == reason, (fields, lm.calls[-1].finish_reason)
                check.equal(got, reason, label=f"finish_reason ({label})")


INVALID = [
    (b"{not json", None),
    (b"", None),
    (b"\xff\xfe{}", None),
    (b"[]", None),
    (b'"hello"', None),
    ({"model": "m"}, "messages"),
    ({"model": "m", "messages": []}, "messages"),
    ({"model": "m", "messages": "hi"}, "messages"),
    ({"model": "m", "messages": ["hi"]}, "messages"),
    ({"model": "m", "messages": [{"role": "user"}]}, "messages"),
    ({"model": "m", "messages": [{"role": "user", "content": 5}]}, "messages"),
    ({"model": "m", "messages": [{"role": "wizard", "content": "hi"}]}, "messages"),
    ({"model": "m", "messages": [{"content": "hi"}]}, "messages"),
    ({"messages": CONVERSATION}, "model"),
    ({"model": 7, "messages": CONVERSATION}, "model"),
    ({"model": "", "messages": CONVERSATION}, "model"),
    ({"model": "m", "messages": CONVERSATION, "max_tokens": 0}, "max_tokens"),
    ({"model": "m", "messages": CONVERSATION, "max_tokens": "3"}, "max_tokens"),
    ({"model": "m", "messages": CONVERSATION, "max_tokens": 2.5}, "max_tokens"),
    ({"model": "m", "messages": CONVERSATION, "max_tokens": True}, "max_tokens"),
    ({"model": "m", "messages": CONVERSATION, "max_completion_tokens": -1}, "max_completion_tokens"),
    ({"model": "m", "messages": CONVERSATION, "stream": "yes"}, "stream"),
    ({"model": "m", "messages": CONVERSATION, "stream": 1}, "stream"),
    ({"model": "m", "messages": CONVERSATION, "stream": True, "stream_options": []}, "stream_options"),
    (
        {"model": "m", "messages": CONVERSATION, "stream": True,
         "stream_options": {"include_usage": "yes"}},
        "stream_options",
    ),
]  # fmt: skip


@case("invalid requests get 400 with an OpenAI-shaped error, and the model is not called",
      public=True)  # fmt: skip
def invalid_requests(ctx):
    # Rejects missing_messages_500 and flat_error_body.
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        for body, param in INVALID:
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            label = f"POST /v1/chat/completions with body {raw[:120]!r}"
            headers = {"content-type": "application/json"}
            response = send(client, "POST", URL, label, content=raw, headers=headers)
            check_error(response, 400, label, param)
        check.that(
            lm.calls == [],
            "the model was called for an invalid request",
            f"prompts: {[call.prompt for call in lm.calls]!r}\n",
        )
        # The server is still healthy, and unknown fields are ignored.
        body = body_of(temperature=0.2, user="u-1", n=1, stream_options=None, stream=None)
        complete(client, lm, body, "a valid request after the invalid ones")


@case("streaming without include_usage: no usage chunk, and usage is never set")
def stream_without_usage(ctx):
    # Rejects usage_chunk_always.
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        first = stream(client, lm, body_of(stream=True), "stream, no stream_options")
        options = {"include_usage": False}
        stream(client, lm, body_of(stream=True, stream_options=options), "include_usage false")
        stream(client, lm, body_of(stream=True, stream_options={}), "empty stream_options")
        second = stream(client, lm, body_of(stream=True, stream_options=None), "null options")
        # stream_options has no effect on a response that is not streamed.
        third = complete(
            client, lm, body_of(stream_options={"include_usage": True}), "options, no stream"
        )
    ids = {first[0]["id"], second[0]["id"], third["id"]}
    check.that(len(ids) == 3, "every completion needs its own id", f"ids: {sorted(ids)}\n")


@case("each piece is flushed before the next one is generated")
def flushes_each_piece(ctx):
    # Rejects buffers_stream. The gate lets one piece through at a time; a server that
    # holds chunks back never delivers the first one, and the model gives up.
    pieces = ["The", " quick", " brown", " fox"]
    lm = ScriptedLM(pieces, gated=True, gate_timeout_s=1.5)
    body = body_of(stream=True, stream_options={"include_usage": True})
    with serve(ctx, lm) as client, client.stream("POST", URL, json=body) as response:
        events = iter_sse(response)
        received = []
        for i, piece in enumerate(pieces):
            lm.release(1)
            # Skip the role chunk; the next chunk with text must be this piece.
            while True:
                event = next_event(events, f"the chunk for piece {i}")
                check.that(
                    not lm.stalled,
                    "chunks are not flushed as pieces are generated: a chunk reached the "
                    "client only after the model had given up waiting",
                    "Send every chunk in its own http.response.body message with "
                    "more_body=True, as soon as its piece is available.\n",
                )
                chunk = event.json()
                received.append(event)
                choices = chunk.get("choices") or [{}]
                if (choices[0].get("delta") or {}).get("content"):
                    break
            check.equal(choices[0]["delta"]["content"], piece, label=f"content of chunk {i}")
            check.equal(
                len(lm.calls[0].pieces), i + 1, label="pieces generated when the chunk arrived"
            )
        lm.release()
        received.extend(events)
    check_chunks(received, lm, 0, body, "a stream released one piece at a time")


@case("generation stops when the client disconnects mid-stream")
def client_disconnect(ctx):
    # Rejects ignores_disconnect. The model would take about a second to run dry.
    pieces = [f" w{i}" for i in range(200)]
    lm = ScriptedLM(pieces, delay_s=0.005)
    with serve(ctx, lm) as client:
        with client.stream("POST", URL, json=body_of(stream=True)) as response:
            events = iter_sse(response)
            for i in range(4):
                next_event(events, f"chunk {i}")
        # Leaving the block closed the connection with the stream unfinished.
        closed = lm.wait_closed(timeout_s=5.0)
        check.that(
            closed,
            "after the client disconnected, the model's stream was left open",
            "Close it with `await pieces.aclose()` when you stop streaming.\n",
        )
        check.that(
            len(lm.calls[0].pieces) < len(pieces),
            "the server kept generating after the client had disconnected",
            f"The client read 4 chunks and closed the connection; the server went on to "
            f"pull all {len(pieces)} pieces from the model.\n"
            "Watch receive() for the http.disconnect message while streaming.\n",
        )
        # The server keeps working for the next client.
        lm.script, lm.delay_s = ["ok"], 0.0
        index = len(lm.calls)
        response = send(client, "POST", URL, "a request after a disconnect", json=body_of())
        check.equal(response.status_code, 200, label="status after another client disconnected")
        check.equal(len(lm.calls), index + 1, label="model calls")


@case("unknown paths get 404, other methods 405, and the query string is ignored")
def routing(ctx):
    lm = ScriptedLM(HELLO)
    with serve(ctx, lm) as client:
        for method, path in [
            ("POST", "/v1/chat/completion"),
            ("POST", "/v1/chat/completions/extra"),
            ("POST", "/v1/completions"),
            ("GET", "/"),
            ("GET", "/v1/models"),
        ]:
            response = send(client, method, path, f"{method} {path}", json=body_of())
            check_error(response, 404, f"{method} {path}")
        for method in ["GET", "PUT", "DELETE"]:
            response = send(client, method, URL, f"{method} {URL}", json=body_of())
            check_error(response, 405, f"{method} {URL}")
            allow = response.headers.get("allow", "")
            check.that(
                "POST" in [m.strip() for m in allow.split(",")],
                f"{method} {URL}: the 405 response has no `allow: POST` header",
                f"allow header: {allow!r}\n",
            )
        check.that(lm.calls == [], "the model was called for a request that was not routed")
        response = send(client, "POST", URL + "?api-version=1", "POST with a query", json=body_of())
        check.equal(response.status_code, 200, label="status of POST with a query string")


@case("conversations, awkward text, and an empty generation, streamed and not")
def conversations(ctx):
    # Rejects usage_in_characters on long prompts and exercises JSON escaping.
    long_chat = [
        {"role": "system", "content": "Answer in one line.\nNever apologize."},
        {"role": "user", "content": 'What does "TTFT" mean?'},
        {"role": "assistant", "content": "Time to first token."},
        {"role": "user", "content": "And   TPOT?  日本語で。"},
    ]
    one_turn = [{"role": "user", "content": ""}]
    scripts = {
        render_prompt(long_chat): ['"Time', " per\noutput", " token\\", "🙂", " é", "\t", "[DONE]"],
        render_prompt(one_turn): [],
        None: ["fallback"],
    }
    lm = ScriptedLM(scripts)
    with serve(ctx, lm) as client:
        for messages in (long_chat, one_turn, CONVERSATION):
            for fields in ({}, {"max_tokens": 3}):
                label = f"{len(messages)} messages, {fields!r}"
                complete(client, lm, body_of(messages, model="org/model-7b", **fields), label)
                options = {"include_usage": True}
                body = body_of(messages, stream=True, stream_options=options, **fields)
                stream(client, lm, body, label + ", streamed")
