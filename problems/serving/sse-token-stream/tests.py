"""Tests for sse-token-stream.

Every case builds the app with `make_app(source)`, where `source` is a scripted token
generator defined here, serves it on a real local socket, and talks to it over HTTP.
The source runs inside the server's event loop and records what the app does with it,
which lets the tests reason about order ("token 1 was not generated before event 0
reached the client") without measuring wall-clock time. All waits are bounded.
"""

import asyncio
import json
import threading
import time

import httpx

from learnai.judge import CheckFailure, case, check, iter_sse, parse_sse, read_sse

URL = "/v1/generate"
GATE_TIMEOUT_S = 2.0  # how long a gated source waits for the test to confirm an event
DISCONNECT_TOKENS = 200
DISCONNECT_INTERVAL_S = 0.005


class Source:
    """A scripted `token_source`: yields the given tokens and records how it is used.

    gated: before generating token i (i >= 1), wait until the test has confirmed, with
        `confirm()`, that it received the event for token i - 1. If no confirmation
        comes within GATE_TIMEOUT_S the source sets `stalled` and ends the generation,
        so a server that holds tokens back cannot hang the test.
    interval_s: pause before each token, as a model would have.
    """

    def __init__(self, tokens, *, gated=False, interval_s=0.0):
        self.tokens = list(tokens)
        self.gated = gated
        self.interval_s = interval_s
        self.calls = []  # (prompt, max_tokens) of every call
        self.produced = 0  # tokens generated so far
        self.confirmed = 0  # events the test has received so far
        self.stalled = False
        self.finished = threading.Event()  # set when a generator ends or is closed

    def __call__(self, prompt, max_tokens):
        self.calls.append((prompt, max_tokens))
        return self._generate()

    def confirm(self):
        self.confirmed += 1

    async def _generate(self):
        try:
            for i, token in enumerate(self.tokens):
                if self.gated and i > 0 and not await self._confirmed(i):
                    self.stalled = True
                    return
                await asyncio.sleep(self.interval_s)
                self.produced += 1
                yield token
        finally:
            self.finished.set()

    async def _confirmed(self, count):
        deadline = time.monotonic() + GATE_TIMEOUT_S
        while self.confirmed < count:
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.001)
        return True


def serve(ctx, source):
    return ctx.serve(ctx.solution.make_app(source))


def next_event(events, what):
    """The next event of a live stream, with bounded waiting and readable failures."""
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


def request(client, method, path, label, **kwargs):
    """A complete (non-streaming) exchange; a stalled or broken response fails the case."""
    try:
        return client.request(method, path, **kwargs)
    except httpx.TimeoutException:
        raise CheckFailure(
            f"{label}: timed out waiting for the complete response",
            "Every response must end: the last http.response.body message of a response "
            "has more_body=False.\n",
        ) from None
    except httpx.HTTPError as exc:
        raise CheckFailure(
            f"{label}: the connection broke: {type(exc).__name__}", f"{exc}\n"
        ) from None


def check_event_stream_response(response):
    """Status 200 and the text/event-stream content type; returns the parsed events."""
    check.equal(response.status_code, 200, label="status of the stream response")
    content_type = response.headers.get("content-type", "")
    check.that(
        content_type.split(";")[0].strip().lower() == "text/event-stream",
        "the content type of the stream is not text/event-stream",
        f"content-type: {content_type!r}\n",
    )
    try:
        text = response.content.decode("utf-8")
    except UnicodeDecodeError:
        raise CheckFailure("the event stream is not valid UTF-8") from None
    return text, list(parse_sse(text.splitlines(keepends=True)))


def check_stream(events, tokens, label):
    """Events must be one JSON event per token, in order, then the [DONE] event."""
    shown = "\n".join(f"  {e.data!r}" for e in events[:12]) or "  (none)"
    detail = f"request: {label}\nexpected tokens: {tokens!r}\nevents received (data):\n{shown}\n"
    check.that(
        bool(events) and events[-1].data == "[DONE]",
        f"{label}: the stream does not end with the event `data: [DONE]`",
        detail,
    )
    check.that(
        len(events) == len(tokens) + 1,
        f"{label}: wrong number of events (one per token plus [DONE] is expected)",
        detail,
    )
    for i, (event, token) in enumerate(zip(events, tokens, strict=False)):
        check.that(
            event.event == "message",
            f"{label}: events must not set an event type",
            detail,
        )
        try:
            payload = json.loads(event.data)
        except ValueError:
            payload = None
        check.that(
            payload == {"token": token, "index": i},
            f'{label}: event {i} is not the JSON object {{"token": ..., "index": {i}}} '
            "for its token",
            detail + f"event {i}: expected {json.dumps({'token': token, 'index': i})}\n",
        )


def check_error(response, status, label):
    detail = (
        f"request: {label}\nstatus: {response.status_code}\n"
        f"content-type: {response.headers.get('content-type')!r}\nbody: {response.text[:200]!r}\n"
    )
    check.that(
        response.status_code == status,
        f"{label}: got status {response.status_code}, expected {status}",
        detail,
    )
    check.that(
        response.headers.get("content-type", "").startswith("application/json"),
        f"{label}: the error response is not application/json",
        detail,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    check.that(
        isinstance(payload, dict)
        and isinstance(payload.get("error"), str)
        and payload["error"] != "",
        f'{label}: the error body is not a JSON object {{"error": "<message>"}}',
        detail,
    )


@case("the worked example: four tokens, then [DONE]", public=True)
def worked_example(ctx):
    tokens = ["Hello", ",", " world", "!"]
    source = Source(tokens)
    with serve(ctx, source) as client:
        events = read_sse(client, "POST", URL, json={"prompt": "Say hello", "max_tokens": 4})
    check_stream(events, tokens, "POST /v1/generate")
    check.equal(source.calls, [("Say hello", 4)], label="token_source calls (prompt, max_tokens)")


@case("each token is flushed before the next one is generated", public=True)
def flushes_each_token(ctx):
    # Rejects buffers_then_sends. The source generates token i + 1 only after the test
    # has received the event for token i, so a server that holds events back stalls it.
    tokens = ["The", " quick", " brown", " fox", " jumps"]
    source = Source(tokens, gated=True)
    body = {"prompt": "go", "max_tokens": 5}
    with serve(ctx, source) as client, client.stream("POST", URL, json=body) as response:
        events = iter_sse(response)
        for i, token in enumerate(tokens):
            event = next_event(events, f"the event for token {i}")
            check.that(
                not source.stalled,
                "tokens are not flushed as they are generated: an event reached the "
                "client only after the generation had moved on or ended",
                f"The event for token {source.confirmed} did not arrive within "
                f"{GATE_TIMEOUT_S:g} s of the token being generated.\n"
                "Send each event in its own http.response.body message with "
                "more_body=True, as soon as its token is available.\n",
            )
            check.equal(event.json(), {"token": token, "index": i}, label=f"event {i}")
            source.confirm()
        last = next_event(events, "the [DONE] event")
        check.equal(last.data, "[DONE]", label="the event after the last token")


@case("a malformed body gets 400 with a JSON error, and nothing is generated", public=True)
def bad_requests(ctx):
    # Rejects accepts_bad_input and unhandled_bad_json.
    source = Source(["x"])
    bodies = [
        b"{not json",
        b"",
        b"\xff\xfe\x00",
        b"[]",
        b'"just a string"',
        b"{}",
        b'{"prompt": 5}',
        b'{"prompt": null}',
        b'{"prompt": "hi", "max_tokens": 0}',
        b'{"prompt": "hi", "max_tokens": -3}',
        b'{"prompt": "hi", "max_tokens": "3"}',
        b'{"prompt": "hi", "max_tokens": 2.5}',
        b'{"prompt": "hi", "max_tokens": true}',
    ]
    with serve(ctx, source) as client:
        for body in bodies:
            label = f"POST /v1/generate with body {body!r}"
            headers = {"content-type": "application/json"}
            response = request(client, "POST", URL, label, content=body, headers=headers)
            check_error(response, 400, label)
        check.that(
            source.calls == [],
            "token_source was called for an invalid request",
            f"calls (prompt, max_tokens): {source.calls!r}\n",
        )
        # The server is still healthy afterwards.
        events = read_sse(client, "POST", URL, json={"prompt": "hi"})
        check_stream(events, ["x"], "a valid request after the invalid ones")


@case("unknown paths get 404 and other methods on /v1/generate get 405", public=True)
def routing(ctx):
    # Rejects wrong_method_is_404.
    source = Source(["x"])
    valid = {"prompt": "hi"}
    with serve(ctx, source) as client:
        for method, path in [
            ("GET", "/"),
            ("POST", "/v1/generat"),
            ("POST", "/v1/generate/"),
            ("POST", "/v1/generate/extra"),
            ("GET", "/v2/generate"),
            ("DELETE", "/nope"),
        ]:
            response = request(client, method, path, f"{method} {path}", json=valid)
            check_error(response, 404, f"{method} {path}")
        for method in ["GET", "PUT", "DELETE", "PATCH"]:
            response = request(client, method, URL, f"{method} /v1/generate", json=valid)
            check_error(response, 405, f"{method} /v1/generate")
            allow = response.headers.get("allow", "")
            check.that(
                "POST" in [m.strip() for m in allow.split(",")],
                f"{method} /v1/generate: the 405 response has no `allow: POST` header",
                f"allow header: {allow!r}\n",
            )
        # The query string is not part of the path.
        events = read_sse(client, "POST", URL + "?stream=true", json=valid)
        check_stream(events, ["x"], "POST /v1/generate?stream=true")
        check.equal(len(source.calls), 1, label="number of token_source calls")


@case("tokens with quotes, newlines, and non-ASCII text survive the wire format")
def awkward_tokens(ctx):
    # Rejects no_json_escape, missing_blank_line, and wrong_content_type.
    tokens = ['"quoted"', "line\nbreak", "\r\n", "back\\slash", "", " ", "日本語", "🙂", "é"]
    tokens += ["data: x", ": colon", "\n\n", "</s>", "[DONE]", "tab\tend"]
    source = Source(tokens)
    with serve(ctx, source) as client:
        body = {"prompt": "ünïcödé \n prompt", "max_tokens": 64}
        response = request(client, "POST", URL, "POST /v1/generate", json=body)
        text, events = check_event_stream_response(response)
        check_stream(events, tokens, "a stream of awkward tokens")
        check.that(
            text.endswith(("\n\n", "\r\r", "\r\n\r\n")),
            "the [DONE] event is not terminated by a blank line",
            f"the body ends with {text[-20:]!r}\n",
        )
    check.equal(source.calls, [("ünïcödé \n prompt", 64)], label="token_source calls")


@case("default max_tokens, extra fields, an empty generation, and a long one")
def request_variants(ctx):
    # Rejects no_done_event (an empty generation is only the [DONE] event).
    long = [f"tok{i}" for i in range(300)]
    source = Source(long)
    with serve(ctx, source) as client:
        body = {"prompt": "p", "temperature": 0.7, "stream": True, "user": {"id": 7}}
        events = read_sse(client, "POST", URL, json=body)
        check_stream(events, long, "a request without max_tokens and with extra fields")
        events = read_sse(client, "POST", URL, json={"max_tokens": 1, "prompt": "q"})
        check_stream(events, long, "a second request on the same server")
        source.tokens = []
        events = read_sse(client, "POST", URL, json={"prompt": ""})
        check_stream(events, [], "a generation of zero tokens")
    check.equal(source.calls, [("p", 16), ("q", 1), ("", 16)], label="token_source calls")


@case("generation stops when the client disconnects mid-stream")
def client_disconnect(ctx):
    # Rejects ignores_disconnect. The source would take about a second to run dry.
    tokens = [f"t{i}" for i in range(DISCONNECT_TOKENS)]
    source = Source(tokens, interval_s=DISCONNECT_INTERVAL_S)
    with serve(ctx, source) as client:
        with client.stream("POST", URL, json={"prompt": "go", "max_tokens": 200}) as response:
            events = iter_sse(response)
            for i in range(3):
                event = next_event(events, f"the event for token {i}")
                check.equal(event.json(), {"token": tokens[i], "index": i}, label=f"event {i}")
        # Leaving the block closed the connection with the stream unfinished.
        ended = source.finished.wait(timeout=5.0)
        check.that(
            ended,
            "after the client disconnected, the token generator was left suspended: it was "
            "neither closed nor run to its end",
            "Close it with `await tokens.aclose()` when you stop streaming.\n",
        )
        check.that(
            source.produced < len(tokens),
            "the app kept generating tokens after the client had disconnected",
            f"The client read 3 events and closed the connection; the app went on to pull "
            f"all {len(tokens)} tokens from the source.\n"
            "Watch receive() for the http.disconnect message while streaming.\n",
        )
        # The server keeps working for the next client.
        source.tokens, source.interval_s = ["ok"], 0.0
        events = read_sse(client, "POST", URL, json={"prompt": "again"})
        check_stream(events, ["ok"], "a request after another client disconnected")
