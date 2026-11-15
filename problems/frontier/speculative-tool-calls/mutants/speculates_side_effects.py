# BUG: asks the predictor for every known tool and starts the predicted call at once, also for a tool with side effects: an email is sent, a table booked, before the model has finished asking for it.
import asyncio
import json

from whetstone.kit import assemble


class Call:
    """One tool call of the turn that is being streamed."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.text = ""  # the argument text received so far
        self.task = None  # the tool run whose result this call will use, if any
        self.guess = None  # the arguments of a speculative run (task was started early)
        self.error = None  # an outcome decided without running anything


async def run_agent(model, tools: dict, messages: list[dict], predictor, max_steps: int) -> list:
    transcript = list(messages)
    for _ in range(max_steps):
        message, outcomes = await run_turn(model.stream(transcript), tools, predictor)
        transcript.append(message)
        calls = message.get("tool_calls")
        if not calls:
            return transcript
        for call, outcome in zip(calls, outcomes, strict=True):
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})
    return transcript


async def run_turn(stream, tools: dict, predictor) -> tuple[dict, list[str]]:
    """Read one streamed turn, starting tools as early as the rules allow."""
    pieces, calls = [], []
    at_turn_end = []  # (call, tool, arguments) of calls to tools with side effects

    def speculate(call: Call) -> None:
        tool = tools.get(call.name)
        if tool is None or call.task is not None:
            return  # nothing to run, not safe to run early, or already running
        guess = predictor(call.name, call.text)
        if guess is not None:
            call.guess, call.task = guess, asyncio.ensure_future(tool(**guess))

    def complete(call: Call) -> None:
        """The call's arguments are final: confirm, replace, or start its run."""
        tool = tools.get(call.name)
        arguments = parse(call.text)
        if call.task is not None and (arguments is None or call.guess != arguments):
            discard(call.task)  # mispredicted: this run must not be used
            call.task = None
        if tool is None:
            call.error = error(f"unknown tool: {call.name}")
        elif arguments is None:
            call.error = error("invalid arguments")
        elif call.task is None:
            if tool.side_effects:
                at_turn_end.append((call, tool, arguments))
            else:
                call.task = asyncio.ensure_future(tool(**arguments))

    async for piece in stream:
        pieces.append(piece)
        for part in piece.get("tool_calls", ()):
            function = part.get("function", {})
            if part["index"] == len(calls):  # a header: the call before it is complete
                if calls:
                    complete(calls[-1])
                calls.append(Call(function["name"]))
            calls[-1].text += function.get("arguments", "")
            speculate(calls[-1])
        if "finish_reason" in piece:  # the end of the turn
            if calls:
                complete(calls[-1])
            for call, tool, arguments in at_turn_end:
                call.task = asyncio.ensure_future(tool(**arguments))

    return assemble(pieces), [await outcome_of(call) for call in calls]


def discard(task) -> None:
    """Drop a speculative run: cancel it, or swallow the failure it already had."""
    if not task.cancel():  # False: it has finished, and may hold an exception
        task.exception()  # retrieved, so asyncio does not report it as lost


async def outcome_of(call: Call) -> str:
    if call.error is not None:
        return call.error
    try:
        result = await call.task
    except Exception as exc:  # noqa: BLE001  a tool may raise anything
        return error(f"tool failed: {exc}")
    return result if isinstance(result, str) else json.dumps(result)


def parse(text: str) -> dict | None:
    """The arguments as a dict, or None if the text is not a JSON object."""
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def error(text: str) -> str:
    return json.dumps({"error": text})
