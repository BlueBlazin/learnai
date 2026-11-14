"""ScriptedLM: a language model that says what the test tells it to, for service problems.

A server is tested by what it does with its model: does it stream each piece as it
arrives, stop asking when the client leaves, count tokens correctly? ``ScriptedLM``
is a model whose output is fixed in advance and which records how it was used:

    lm = ScriptedLM(["Hello", ",", " world", "!"])
    async for piece in lm.stream(prompt, max_tokens=3):     # "Hello", ",", " world"
        ...

Each piece of the script counts as one generated token. ``stream`` yields at most
``max_tokens`` pieces. As with a real model, a generation that produced exactly
``max_tokens`` pieces was cut off by the limit (finish reason ``length``); one that
ended earlier ended by itself (``stop``). ``lm.count_tokens(text)`` is the number of
whitespace-separated words, the model's notion of prompt tokens.

The stream returned by ``stream`` is an async iterator with ``aclose()``. It is
*closed* once it has yielded everything, once ``aclose()`` has been called, or once an
exception or a cancellation has interrupted it while it was producing a piece. A
server that stops reading without closing it leaves it open, and ``lm.closed`` false.

The gate lets a test decide when pieces may be produced, to prove that a server
streams instead of buffering, without measuring time:

    lm = ScriptedLM(script, gated=True)     # nothing is produced until released
    lm.release(1)                            # allow one piece
    lm.release()                             # open the gate for good

``release`` may be called from the test's thread while the server's event loop runs
the stream. A gated stream that is not released within ``gate_timeout_s`` gives up:
it ends early and sets ``lm.stalled``, so a wrong server fails the test instead of
hanging it.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from whetstone.kit.counters import Counters

Script = Sequence[str] | Mapping[str, Sequence[str]] | Callable[[str], Sequence[str]]


@dataclass
class StreamRecord:
    """How one call of ``stream`` went.

    Attributes:
        prompt, max_tokens: the arguments of the call.
        script: the pieces this call could produce (before ``max_tokens``).
        pieces: the pieces produced so far.
        closed: whether the stream has ended or been closed.
        finish_reason: ``"stop"`` or ``"length"`` once every piece allowed has been
            produced; None before that, and for a stream closed early.
    """

    prompt: str
    max_tokens: int | None
    script: tuple[str, ...]
    pieces: list[str] = field(default_factory=list)
    closed: bool = False

    @property
    def limit(self) -> int:
        """How many pieces the stream produces if it runs to its end."""
        return len(self.script) if self.max_tokens is None else min(len(self.script), self.max_tokens)

    @property
    def finish_reason(self) -> str | None:
        if len(self.pieces) < self.limit:
            return None
        return "length" if len(self.pieces) == self.max_tokens else "stop"


class ScriptedLM:
    """A scripted token source. See the module docstring.

    Args:
        script: what to generate: a list of pieces used for every prompt; a dict from
            prompt to pieces (the key ``None``, if present, is the default; without
            it, ``stream`` raises ``KeyError`` for a prompt that is not listed); or a
            function from prompt to pieces.
        gated: start with the gate shut (see ``release``).
        delay_s: pause before each piece, in the event loop's time (virtual under a
            simulated clock).
        gate_timeout_s: how long a gated stream waits for ``release``.

    Attributes:
        calls: one ``StreamRecord`` per call of ``stream``, in call order.
        active: streams started and not yet closed.
        closed: True when at least one stream was started and none is open.
        stalled: True if a gated stream gave up waiting.
        counters: ``streams`` and ``pieces``.
    """

    def __init__(
        self,
        script: Script,
        *,
        gated: bool = False,
        delay_s: float = 0.0,
        gate_timeout_s: float = 5.0,
    ) -> None:
        self.script = script
        self.delay_s = delay_s
        self.gate_timeout_s = gate_timeout_s
        self.calls: list[StreamRecord] = []
        self.stalled = False
        self.counters = Counters("streams", "pieces")
        self._lock = threading.Lock()
        self._permits: int | None = 0 if gated else None  # None: the gate is open
        self._idle = threading.Event()

    # -- the model -----------------------------------------------------------------------

    def stream(self, prompt: str, max_tokens: int | None = None) -> ScriptedStream:
        """Start generating for ``prompt``; iterate the result with ``async for``.

        ``max_tokens`` is None (no limit) or an int >= 0, else ``ValueError``.
        """
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0
        ):
            raise ValueError(f"max_tokens must be None or an int >= 0, got {max_tokens!r}")
        record = StreamRecord(prompt, max_tokens, tuple(self._pieces_for(prompt)))
        with self._lock:
            self.calls.append(record)
            self._idle.clear()
        self.counters.add("streams")
        return ScriptedStream(self, record)

    def count_tokens(self, text: str) -> int:
        """The number of tokens in ``text``: its whitespace-separated words."""
        return len(text.split())

    # -- what the test controls and observes ---------------------------------------------

    def release(self, n: int | None = None) -> None:
        """Let gated streams produce ``n`` more pieces; with no argument, open the gate."""
        with self._lock:
            if n is None:
                self._permits = None
            elif self._permits is not None:
                self._permits += n

    @property
    def active(self) -> int:
        return sum(not record.closed for record in self.calls)

    @property
    def closed(self) -> bool:
        return bool(self.calls) and self.active == 0

    def wait_closed(self, timeout_s: float = 2.0) -> bool:
        """Block the calling thread until every stream is closed; False on timeout.

        For tests, whose thread is not the server's: after closing a connection, wait
        here for the server to notice, then assert.
        """
        return self._idle.wait(timeout_s) and self.closed

    # -- internals -----------------------------------------------------------------------

    def _pieces_for(self, prompt: str) -> Sequence[str]:
        if callable(self.script):
            return self.script(prompt)
        if isinstance(self.script, Mapping):
            return self.script[prompt] if prompt in self.script else self.script[None]
        return self.script

    def _take_permit(self) -> bool:
        with self._lock:
            if self._permits is None:
                return True
            if self._permits > 0:
                self._permits -= 1
                return True
            return False

    def _close(self, record: StreamRecord) -> None:
        with self._lock:
            record.closed = True
            if all(r.closed for r in self.calls):
                self._idle.set()


class ScriptedStream:
    """The async iterator ``ScriptedLM.stream`` returns. ``record`` describes its progress."""

    def __init__(self, lm: ScriptedLM, record: StreamRecord) -> None:
        self._lm = lm
        self.record = record

    def __aiter__(self) -> ScriptedStream:
        return self

    async def __anext__(self) -> str:
        record = self.record
        if record.closed or len(record.pieces) >= record.limit:
            self._lm._close(record)
            raise StopAsyncIteration
        try:
            if not await self._gate_opens():
                self._lm.stalled = True
                self._lm._close(record)
                raise StopAsyncIteration
            await _pause(self._lm.delay_s)
        except BaseException:
            # Interrupted while producing a piece (a cancellation, usually): the
            # generation is over.
            self._lm._close(record)
            raise
        piece = record.script[len(record.pieces)]
        record.pieces.append(piece)
        self._lm.counters.add("pieces")
        if len(record.pieces) >= record.limit:
            self._lm._close(record)  # nothing left to produce
        return piece

    async def aclose(self) -> None:
        """Stop generating. Safe to call more than once, and on a finished stream."""
        self._lm._close(self.record)

    async def _gate_opens(self) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._lm.gate_timeout_s
        while not self._lm._take_permit():
            if loop.time() >= deadline:
                return False
            await _pause(0.001)
        return True


async def _pause(delay_s: float) -> None:
    """Let the event loop run for ``delay_s`` (at least one turn).

    A timer on the loop rather than ``asyncio.sleep``, so it also works in problems
    whose rules forbid the solution to sleep.
    """
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    handle = loop.call_later(delay_s, lambda: done.done() or done.set_result(None))
    try:
        await done
    finally:
        handle.cancel()
