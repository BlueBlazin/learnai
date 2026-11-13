"""Service judge: run the solution's ASGI app for real and talk to it over HTTP.

The app is served by uvicorn on ``127.0.0.1`` with an ephemeral port (the OS picks a
free one), in a background thread of the harness process. The test gets an
``httpx.Client`` bound to that server. All waiting is bounded: the client has a
timeout, and the server is always shut down when the ``with`` block exits.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from whetstone.judge.errors import CheckFailure

STARTUP_TIMEOUT_S = 10.0
SHUTDOWN_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class SSEEvent:
    """One server-sent event, per the WHATWG EventSource parsing rules."""

    data: str
    event: str = "message"
    id: str | None = None
    retry: int | None = None

    def json(self) -> Any:
        """The data field parsed as JSON (fails the case if it is not JSON)."""
        try:
            return json.loads(self.data)
        except ValueError:
            raise CheckFailure(
                "an event's data is not valid JSON", f"data: {self.data!r}\n"
            ) from None


def parse_sse(lines: Iterable[str]) -> Iterator[SSEEvent]:
    """Parse an iterable of text lines into server-sent events.

    Follows the standard: ``field: value`` lines (one optional space after the colon),
    lines starting with ``:`` are comments, several ``data`` lines join with newlines,
    and a blank line dispatches the event. An unterminated final event is discarded,
    as a browser would.
    """
    data: list[str] = []
    event, last_id, retry = "message", None, None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                yield SSEEvent("\n".join(data), event, last_id, retry)
            data, event, retry = [], "message", None
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value or "message"
        elif name == "id":
            last_id = value
        elif name == "retry" and value.isdigit():
            retry = int(value)


def iter_sse(response: httpx.Response) -> Iterator[SSEEvent]:
    """Iterate over the events of a streaming response as they arrive.

    Use inside ``with client.stream("GET", url) as response:``. Checks the status code
    and the ``text/event-stream`` content type first.
    """
    if response.status_code != 200:
        raise CheckFailure(f"expected status 200 for the event stream, got {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("text/event-stream"):
        raise CheckFailure(
            "the response is not a server-sent event stream "
            f"(content-type is {content_type or 'missing'!r}, expected 'text/event-stream')"
        )
    yield from parse_sse(response.iter_lines())


def read_sse(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_events: int | None = None,
    stop_at: str | None = None,
    **request: Any,
) -> list[SSEEvent]:
    """Make a request and collect its server-sent events.

    Reads until the server closes the stream, ``max_events`` have arrived, or an event
    whose data equals ``stop_at`` arrives (that event is included), for example
    ``stop_at="[DONE]"``. Bounded by the client's timeout.
    """
    events: list[SSEEvent] = []
    try:
        with client.stream(method, url, **request) as response:
            for event in iter_sse(response):
                events.append(event)
                if max_events is not None and len(events) >= max_events:
                    break
                if stop_at is not None and event.data == stop_at:
                    break
    except httpx.TimeoutException:
        raise CheckFailure(
            f"timed out reading the event stream from {url} after {len(events)} events",
            "The stream must end (or send the terminating event) once generation is done.\n",
        ) from None
    except httpx.HTTPError as exc:
        raise CheckFailure(
            f"the event stream from {url} broke after {len(events)} events: {type(exc).__name__}",
            f"{exc}\n",
        ) from None
    return events


class AppServer:
    """A uvicorn server for one ASGI app on an ephemeral local port."""

    def __init__(self, app: Any) -> None:
        import uvicorn

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,  # let the OS choose a free port
            log_level="warning",
            access_log=False,
            lifespan="auto",
            timeout_graceful_shutdown=1,
        )
        self._server = uvicorn.Server(config)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="whetstone-app", daemon=True)
        self.port = 0

    def _run(self) -> None:
        try:
            self._server.run()
        except BaseException as exc:  # noqa: BLE001 - reported from start()
            self._error = exc

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive() or self._error is not None:
                raise CheckFailure(
                    "the app failed to start",
                    f"{type(self._error).__name__}: {self._error}\n" if self._error else None,
                )
            if time.monotonic() > deadline:
                self.stop()
                raise CheckFailure(f"the app did not start within {STARTUP_TIMEOUT_S:g} s")
            time.sleep(0.002)
        self.port = self._server.servers[0].sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(SHUTDOWN_TIMEOUT_S)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(SHUTDOWN_TIMEOUT_S)


@contextmanager
def serve(app: Any, *, timeout_s: float = 5.0) -> Iterator[httpx.Client]:
    """Serve ``app`` and yield an ``httpx.Client`` whose base URL is the server.

    ``timeout_s`` bounds every connect and every read on the client, so a stalled
    endpoint fails the case instead of hanging it. ``client.base_url`` has the address
    if a second client is needed.
    """
    server = AppServer(app)
    server.start()
    try:
        # trust_env=False: ignore proxy settings; this is loopback only.
        with httpx.Client(base_url=server.base_url, timeout=timeout_s, trust_env=False) as client:
            yield client
    finally:
        server.stop()
