# BUG: tool_call_id is built from a running counter (call_0, call_1, ...) instead of being read from the call, so it is wrong whenever the model chooses its own ids.
import asyncio
import json


async def run_agent(model, tools: dict, messages: list[dict], max_steps: int) -> list[dict]:
    transcript = list(messages)
    answered = 0

    for _ in range(max_steps):
        reply = await model.acomplete(transcript)
        transcript.append(reply)
        calls = reply.get("tool_calls")
        if not calls:
            return transcript

        outcomes = await asyncio.gather(*(execute(tools, call) for call in calls))
        for outcome in outcomes:
            transcript.append(
                {"role": "tool", "tool_call_id": f"call_{answered}", "content": outcome}
            )
            answered += 1

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

    try:
        result = await tool(**arguments)
    except Exception as exc:  # noqa: BLE001  a tool may raise anything
        return error(f"tool failed: {exc}")
    return result if isinstance(result, str) else json.dumps(result)


def error(text: str) -> str:
    return json.dumps({"error": text})
