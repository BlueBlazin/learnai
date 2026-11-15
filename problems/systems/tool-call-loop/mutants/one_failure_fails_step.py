# BUG: the calls of a step are gathered without a per-call try, and the first exception is reported as the outcome of every call of the step, including the ones that succeeded.
import asyncio
import json


async def run_agent(model, tools: dict, messages: list[dict], max_steps: int) -> list[dict]:
    transcript = list(messages)

    for _ in range(max_steps):
        reply = await model.acomplete(transcript)
        transcript.append(reply)
        calls = reply.get("tool_calls")
        if not calls:
            return transcript

        try:
            outcomes = await asyncio.gather(*(execute(tools, call) for call in calls))
        except Exception as exc:  # noqa: BLE001  a tool may raise anything
            outcomes = [error(f"tool failed: {exc}")] * len(calls)
        for call, outcome in zip(calls, outcomes, strict=True):
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})

    return transcript


async def execute(tools: dict, call: dict) -> str:
    """The outcome of one tool call. Never raises: an error is a result for the model."""
    function = call["function"]
    tool = tools.get(function["name"])
    if tool is None:
        return error(f"unknown tool: {function['name']}")

    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return error("invalid arguments")
    if not isinstance(arguments, dict):
        return error("invalid arguments")

    result = await tool(**arguments)
    return result if isinstance(result, str) else json.dumps(result)


def error(text: str) -> str:
    return json.dumps({"error": text})
