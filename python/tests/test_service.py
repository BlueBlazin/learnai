import socket
import threading

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from whetstone.judge import CheckFailure, SSEEvent, iter_sse, parse_sse, read_sse, serve


def make_app(gate: threading.Event | None = None) -> Starlette:
    async def health(request):
        return JSONResponse({"ok": True})

    async def stream(request):
        async def events():
            yield "data: one\n\n"
            if gate is not None:
                # Hold the second event until the test has seen the first.
                import asyncio

                while not gate.is_set():
                    await asyncio.sleep(0.005)
            yield "data: two\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    async def stall(request):
        async def events():
            import asyncio

            yield "data: first\n\n"
            await asyncio.sleep(30)

        return StreamingResponse(events(), media_type="text/event-stream")

    async def plain(request):
        return PlainTextResponse("data: nope\n\n")

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/stream", stream),
            Route("/stall", stall),
            Route("/plain", plain),
        ]
    )


def port_is_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def test_parse_sse_follows_the_event_stream_rules():
    lines = [
        ": a comment",
        "data: first",
        "",
        "event: token",
        "id: 7",
        "data:no space",
        "data: second line",
        "",
        "retry: 1500",
        'data: {"token": "hi"}',
        "",
        "data: unterminated",
    ]
    events = list(parse_sse(lines))
    assert events == [
        SSEEvent("first"),
        SSEEvent("no space\nsecond line", event="token", id="7"),
        SSEEvent('{"token": "hi"}', id="7", retry=1500),  # the last id persists
    ]
    assert events[2].json() == {"token": "hi"}
    with pytest.raises(CheckFailure, match="not valid JSON"):
        events[0].json()


def test_serve_runs_the_app_on_an_ephemeral_port_and_shuts_down():
    with serve(make_app()) as client:
        port = client.base_url.port
        assert client.base_url.host == "127.0.0.1"
        assert port not in (None, 0, 8000)
        assert client.get("/health").json() == {"ok": True}
        assert client.get("/missing").status_code == 404
        assert port_is_open(port)
    assert not port_is_open(port)
    assert not [t for t in threading.enumerate() if t.name == "whetstone-app"]


def test_two_servers_get_different_ports():
    with serve(make_app()) as first, serve(make_app()) as second:
        assert first.base_url.port != second.base_url.port
        assert second.get("/health").status_code == 200


def test_read_sse_collects_events():
    with serve(make_app()) as client:
        events = read_sse(client, "GET", "/stream")
        assert [e.data for e in events] == ["one", "two", "[DONE]"]
        assert [e.data for e in read_sse(client, "GET", "/stream", max_events=1)] == ["one"]
        assert [e.data for e in read_sse(client, "GET", "/stream", stop_at="two")] == ["one", "two"]


def test_events_can_be_observed_as_they_arrive():
    gate = threading.Event()
    with serve(make_app(gate)) as client, client.stream("GET", "/stream") as response:
        events = iter_sse(response)
        assert next(events).data == "one"  # arrives while the server is still holding "two"
        gate.set()
        assert [e.data for e in events] == ["two", "[DONE]"]


def test_wrong_content_type_and_status_are_check_failures():
    with serve(make_app()) as client:
        with pytest.raises(CheckFailure, match="not a server-sent event stream"):
            read_sse(client, "GET", "/plain")
        with pytest.raises(CheckFailure, match="expected status 200 for the event stream, got 404"):
            read_sse(client, "GET", "/missing")


def test_a_stalled_stream_times_out_and_the_server_still_stops():
    with serve(make_app(), timeout_s=0.3) as client:
        port = client.base_url.port
        with pytest.raises(CheckFailure, match="timed out reading the event stream"):
            read_sse(client, "GET", "/stall")
    assert not port_is_open(port)


def test_server_is_stopped_when_the_test_body_raises():
    with pytest.raises(RuntimeError), serve(make_app()) as client:
        port = client.base_url.port
        raise RuntimeError("test body failed")
    assert not port_is_open(port)


def test_an_app_that_cannot_start_is_reported():
    async def broken_lifespan(scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "no model"})

    with pytest.raises(CheckFailure, match="the app failed to start"), serve(broken_lifespan):
        pass


def test_client_ignores_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    with serve(make_app()) as client:
        assert client.get("/health").status_code == 200
    assert isinstance(client, httpx.Client)
