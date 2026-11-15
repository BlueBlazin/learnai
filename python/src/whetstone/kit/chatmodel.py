"""ScriptedChatModel and ScriptedTool: a chat model and tools that follow a script.

An agent loop, a router, a speculative tool runner: each is tested by what it does
*around* a model. The model's part has to be fixed in advance, and it has to notice
when it is used wrongly. ``ScriptedChatModel`` is an OpenAI-style chat model whose
replies are scripted:

    model = ScriptedChatModel([
        reply(tool_calls=[tool_call("get_weather", {"city": "Paris"}),
                          tool_call("get_time", {"zone": "CET"})]),      # two calls at once
        reply(tool_calls=[tool_call("get_weather", '{"city": ')]),       # malformed arguments
        "It is 21 degrees in Paris.",                                     # the final answer
    ])
    message = model.complete(messages)

Messages are plain dicts in the chat-completions shape. What ``complete`` returns:

    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}

``arguments`` is a JSON *string*, as on the wire; a script may give a dict (encoded
with ``json.dumps``) or a string, which is passed through untouched, valid or not.
``content`` is a string or None; ``tool_calls`` is absent when there are none; a reply
scripted with ``logprobs=[...]`` carries them as ``message["logprobs"]``. Call ids are
``call_0``, ``call_1``, ... in the order the model emits them, unless the script names
one.

The script
----------
A list of replies, used in order, one per call; or a dict from the content of the last
message of the conversation to a reply (key ``None``: the default); or a function
``(messages, call_index) -> reply``. A reply is a string (the content), the result of
``reply(...)``, or a list of ``tool_call(...)``. A list that has run out raises
``ScriptExhausted`` (a loop without a step limit ends there), unless
``repeat_last=True``.

What the model checks
---------------------
With ``strict=True`` (the default) every call validates the conversation it is given,
as a real API does, and raises ``ChatProtocolError`` with a ``code``:

    bad_message              not a dict with a known role, or a tool message without an id
    missing_tool_result      an assistant message with tool calls is not followed by one
                             tool message for each of its ids
    unknown_tool_call_id     a tool message answers an id the model never issued there
    duplicate_tool_result    an id answered twice

Every error is also kept in ``model.violations``, so a solution that catches the
exception is still caught by the test.

What the test reads
-------------------
``model.calls`` (a deep copy of the messages of every call), ``model.replies`` (what
it answered), ``model.counters`` (``calls``, ``tool_calls``), ``model.cost`` (the price
of one call, fixed at construction) and ``model.spent`` (``cost`` times calls).

Asynchronous use
----------------
``await model.acomplete(messages)`` is ``complete`` after ``delay_s`` seconds of the
event loop's time (virtual under a simulated clock). ``model.stream(messages)`` is an
async iterator over the same reply in pieces, one every ``delay_s``:

    {"content": "It is 21"}                                   content, chunk_chars at a time
    {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": ""}}]}
    {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}    fragments
    {"finish_reason": "tool_calls"}                           or "stop"; always last

``assemble(deltas)`` puts such pieces back together into a message.

``ScriptedTool`` is the other half: a named async function with a latency, a
side-effect flag, and a log of every run that says whether it finished, failed, or
was canceled.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from whetstone.kit.counters import Counters

ROLES = ("system", "user", "assistant", "tool")


class ChatProtocolError(RuntimeError):
    """The conversation handed to the model is not a valid one. ``code`` says why."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ScriptExhausted(RuntimeError):
    """The model was called more often than its script has replies."""


def tool_call(name: str, arguments: Any = None, id: str | None = None) -> dict[str, Any]:
    """One tool call for a scripted reply.

    ``arguments`` is a dict (encoded as JSON), or a string that is passed through as
    it is: ``tool_call("search", '{"q": ')`` scripts a call with malformed arguments.
    ``id`` fixes the call's id; by default the model numbers its calls.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("a tool call needs the tool's name")
    if arguments is None:
        arguments = {}
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"name": name, "arguments": text, "id": id}


def reply(
    content: str | None = None,
    *,
    tool_calls: Iterable[Mapping[str, Any]] = (),
    logprobs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """A scripted reply: text, tool calls (``tool_call(...)``), or both.

    ``logprobs`` are the log-probabilities of the reply's tokens, one per token, for
    problems that judge a model's confidence.
    """
    calls = [dict(call) for call in tool_calls]
    for call in calls:
        if set(call) != {"name", "arguments", "id"}:
            raise ValueError("tool_calls must be built with tool_call(...)")
    if content is not None and not isinstance(content, str):
        raise ValueError("content must be a string or None")
    out: dict[str, Any] = {"content": content, "tool_calls": calls}
    if logprobs is not None:
        out["logprobs"] = [float(value) for value in logprobs]
    return out


def assemble(deltas: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The assistant message a sequence of ``stream`` pieces adds up to."""
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    for delta in deltas:
        if delta.get("content"):
            content.append(delta["content"])
        for part in delta.get("tool_calls", ()):
            call = calls.setdefault(
                part["index"],
                {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if part.get("id"):
                call["id"] = part["id"]
            function = part.get("function", {})
            call["function"]["name"] += function.get("name", "")
            call["function"]["arguments"] += function.get("arguments", "")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return message


class ScriptedChatModel:
    """A chat model that replays a script. See the module docstring.

    Args:
        script: a list of replies, a dict keyed on the last message's content, or a
            function ``(messages, call_index) -> reply``.
        name: the model's name, for problems with several models.
        cost: the price of one call, in whatever unit the problem uses.
        strict: validate every conversation (``ChatProtocolError``).
        repeat_last: when a list script has run out, keep giving its last reply.
        delay_s: time ``acomplete`` takes, and the pause before each piece of ``stream``.
        chunk_chars: how many characters of content or arguments one stream piece carries.
    """

    def __init__(
        self,
        script: Any,
        *,
        name: str = "model",
        cost: float = 0.0,
        strict: bool = True,
        repeat_last: bool = False,
        delay_s: float = 0.0,
        chunk_chars: int = 8,
    ) -> None:
        if chunk_chars < 1:
            raise ValueError("chunk_chars must be at least 1")
        if cost < 0 or delay_s < 0:
            raise ValueError("cost and delay_s must not be negative")
        self.script = script
        self.name = name
        self.cost = cost
        self.strict = strict
        self.repeat_last = repeat_last
        self.delay_s = delay_s
        self.chunk_chars = chunk_chars
        self.calls: list[list[dict[str, Any]]] = []
        self.replies: list[dict[str, Any]] = []
        self.violations: list[ChatProtocolError] = []
        self.counters = Counters("calls", "tool_calls")
        self._next_id = 0

    def with_framework(self, fw: Any) -> ScriptedChatModel:
        """The same script from its start, with empty logs (the oracle's own copy)."""
        twin = object.__new__(ScriptedChatModel)
        twin.__dict__.update(self.__dict__)
        twin.calls, twin.replies, twin.violations = [], [], []
        twin.counters = Counters("calls", "tool_calls")
        twin._next_id = 0
        return twin

    @property
    def spent(self) -> float:
        """What the calls so far have cost: ``cost`` times ``counters.calls``."""
        return self.cost * self.counters.calls

    # -- the model -------------------------------------------------------------------------

    def complete(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The model's next message for the conversation ``messages``."""
        return copy.deepcopy(self._reply(messages))

    async def acomplete(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """``complete``, ``delay_s`` seconds of the event loop's time later."""
        message = self._reply(messages)
        await _pause(self.delay_s)
        return copy.deepcopy(message)

    def stream(self, messages: Sequence[Mapping[str, Any]]) -> ChatStream:
        """The reply in pieces: ``async for delta in model.stream(messages)``."""
        return ChatStream(self, _deltas(self._reply(messages), self.chunk_chars))

    # -- internals ---------------------------------------------------------------------------

    def _reply(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if self.strict:
            self._check(messages)
        index = self.counters.calls
        scripted = self._scripted(messages, index)
        self.counters.add("calls")
        self.calls.append(copy.deepcopy([dict(m) for m in messages]))
        message = self._message(scripted)
        self.counters.add("tool_calls", len(message.get("tool_calls", ())))
        self.replies.append(copy.deepcopy(message))
        return message

    def _scripted(self, messages: Sequence[Mapping[str, Any]], index: int) -> Any:
        script = self.script
        if callable(script):
            return script(messages, index)
        if isinstance(script, Mapping):
            key = messages[-1].get("content") if messages else None
            if key in script:
                return script[key]
            if None in script:
                return script[None]
            raise KeyError(f"the script has no reply for a conversation ending in {key!r}")
        if index < len(script):
            return script[index]
        if self.repeat_last and len(script):
            return script[-1]
        raise ScriptExhausted(
            f"{self.name}: called {index + 1} times, but its script has {len(script)} replies"
        )

    def _message(self, scripted: Any) -> dict[str, Any]:
        if isinstance(scripted, str):
            scripted = reply(scripted)
        elif isinstance(scripted, (list, tuple)):
            scripted = reply(tool_calls=scripted)
        elif isinstance(scripted, Mapping) and "tool_calls" not in scripted:
            scripted = {"tool_calls": [], **scripted}
        elif not isinstance(scripted, Mapping):
            raise TypeError(
                "a scripted reply is a string, reply(...), or a list of tool_call(...); "
                f"got {type(scripted).__name__}"
            )
        message: dict[str, Any] = {"role": "assistant", "content": scripted.get("content")}
        calls = []
        for call in scripted["tool_calls"]:
            call_id = call["id"]
            if call_id is None:
                call_id = f"call_{self._next_id}"
                self._next_id += 1
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
            )
        if calls:
            message["tool_calls"] = calls
        if "logprobs" in scripted:
            message["logprobs"] = list(scripted["logprobs"])
        return message

    def _check(self, messages: Any) -> None:
        if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, Sequence):
            raise self._violation("bad_message", "messages must be a list of message dicts")
        open_ids: dict[str, bool] = {}  # ids of the last assistant turn -> answered?
        for position, message in enumerate(messages):
            if not isinstance(message, Mapping) or message.get("role") not in ROLES:
                raise self._violation(
                    "bad_message",
                    f"message {position} is not a dict with a role in {', '.join(ROLES)}",
                )
            role = message["role"]
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str):
                    raise self._violation(
                        "bad_message", f"message {position}: a tool message needs `tool_call_id`"
                    )
                if call_id not in open_ids:
                    raise self._violation(
                        "unknown_tool_call_id",
                        f"message {position} answers {call_id!r}, which the assistant "
                        "message before it did not call",
                    )
                if open_ids[call_id]:
                    raise self._violation(
                        "duplicate_tool_result",
                        f"message {position}: {call_id!r} is answered twice",
                    )
                open_ids[call_id] = True
                continue
            self._all_answered(open_ids, position)
            open_ids = {}
            if role == "assistant":
                for call in message.get("tool_calls") or ():
                    open_ids[call["id"]] = False
        self._all_answered(open_ids, None)

    def _all_answered(self, open_ids: dict[str, bool], position: int | None) -> None:
        missing = [call_id for call_id, answered in open_ids.items() if not answered]
        if missing:
            where = "the end of the conversation" if position is None else f"message {position}"
            raise self._violation(
                "missing_tool_result",
                f"before {where}: no tool message answers {', '.join(missing)}",
            )

    def _violation(self, code: str, message: str) -> ChatProtocolError:
        error = ChatProtocolError(code, f"{self.name}: {message}")
        self.violations.append(error)
        return error

    def __repr__(self) -> str:
        return f"ScriptedChatModel({self.name!r}, {self.counters!r})"


def _deltas(message: Mapping[str, Any], chunk: int) -> list[dict[str, Any]]:
    """The stream pieces of ``message`` (see the module docstring)."""
    out: list[dict[str, Any]] = []
    content = message.get("content") or ""
    out.extend({"content": content[i : i + chunk]} for i in range(0, len(content), chunk))
    calls = message.get("tool_calls", ())
    for index, call in enumerate(calls):
        function = call["function"]
        out.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": function["name"], "arguments": ""},
                    }
                ]
            }
        )
        text = function["arguments"]
        out.extend(
            {"tool_calls": [{"index": index, "function": {"arguments": text[i : i + chunk]}}]}
            for i in range(0, len(text), chunk)
        )
    out.append({"finish_reason": "tool_calls" if calls else "stop"})
    return out


class ChatStream:
    """The async iterator ``ScriptedChatModel.stream`` returns.

    ``produced`` are the pieces handed out so far; ``closed`` is true once the last
    piece is out, after ``aclose()``, or after a cancellation interrupted it.
    """

    def __init__(self, model: ScriptedChatModel, deltas: list[dict[str, Any]]) -> None:
        self._model = model
        self._deltas = deltas
        self.produced: list[dict[str, Any]] = []
        self.closed = False

    def __aiter__(self) -> ChatStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self.closed or len(self.produced) >= len(self._deltas):
            self.closed = True
            raise StopAsyncIteration
        try:
            await _pause(self._model.delay_s)
        except BaseException:
            self.closed = True
            raise
        delta = self._deltas[len(self.produced)]
        self.produced.append(delta)
        if len(self.produced) == len(self._deltas):
            self.closed = True
        return copy.deepcopy(delta)

    async def aclose(self) -> None:
        self.closed = True


async def _pause(delay_s: float) -> None:
    """Let the event loop run for ``delay_s`` (at least one turn), without ``asyncio.sleep``."""
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    handle = loop.call_later(delay_s, lambda: done.done() or done.set_result(None))
    try:
        await done
    finally:
        handle.cancel()


# -- tools ---------------------------------------------------------------------------------


@dataclass
class ToolRun:
    """One run of a ``ScriptedTool``.

    ``status`` is ``"running"``, then ``"done"``, ``"failed"`` (the function raised), or
    ``"canceled"`` (the awaiting task was canceled before the tool finished).
    ``started_s`` and ``finished_s`` are the event loop's time (virtual under a
    simulated clock); ``finished_s`` is None while running.
    """

    arguments: dict[str, Any]
    started_s: float
    finished_s: float | None = None
    status: str = "running"
    result: Any = None


@dataclass
class ScriptedTool:
    """A tool an agent can call: ``await tool(city="Paris")``.

        weather = ScriptedTool("get_weather", lambda city: f"21 C in {city}", delay_s=0.5)
        charge = ScriptedTool("charge_card", pay, side_effects=True)

    Calling it records a ``ToolRun``, waits ``delay_s`` of the event loop's time, then
    returns ``fn(**arguments)`` (an exception from ``fn`` propagates, after the
    delay). If the awaiting task is canceled meanwhile, the run is marked
    ``"canceled"`` and ``fn`` is never called: a canceled speculative call has no
    effect. ``tool.run(**arguments)`` is the same without the delay, for synchronous
    code.

    ``side_effects`` marks a tool that changes the world (it must never be run
    speculatively); ``parameters`` is its JSON Schema, and ``tool.schema`` the
    ``{"type": "function", "function": {...}}`` entry of a ``tools`` list.
    ``tool.runs`` is the log; ``tool.counters`` has ``started``, ``done``, ``failed``,
    ``canceled``.
    """

    name: str
    fn: Callable[..., Any]
    delay_s: float = 0.0
    side_effects: bool = False
    parameters: dict[str, Any] | None = None
    description: str = ""
    runs: list[ToolRun] = field(default_factory=list)
    counters: Counters = field(
        default_factory=lambda: Counters("started", "done", "failed", "canceled")
    )

    @property
    def schema(self) -> dict[str, Any]:
        parameters = self.parameters or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    async def __call__(self, **arguments: Any) -> Any:
        loop = asyncio.get_running_loop()
        run = self._start(arguments, loop.time())
        try:
            await _pause(self.delay_s)
        except asyncio.CancelledError:
            self._finish(run, "canceled", loop.time())
            raise
        return self._call(run, arguments, loop.time)

    def run(self, **arguments: Any) -> Any:
        """Call the tool synchronously: no delay, times recorded as 0."""
        return self._call(self._start(arguments, 0.0), arguments, lambda: 0.0)

    def _start(self, arguments: dict[str, Any], now: float) -> ToolRun:
        run = ToolRun(copy.deepcopy(arguments), now)
        self.runs.append(run)
        self.counters.add("started")
        return run

    def _call(self, run: ToolRun, arguments: dict[str, Any], clock: Callable[[], float]) -> Any:
        try:
            run.result = self.fn(**arguments)
        except Exception:
            self._finish(run, "failed", clock())
            raise
        self._finish(run, "done", clock())
        return run.result

    def _finish(self, run: ToolRun, status: str, now: float) -> None:
        run.status, run.finished_s = status, now
        self.counters.add(status)
