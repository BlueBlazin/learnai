import asyncio
import json


async def run_agent(model, tools: dict, messages: list[dict], max_steps: int) -> list[dict]:
    transcript = list(messages)  # the caller's list is not ours to grow

    for _ in range(max_steps):
        reply = await model.acomplete(transcript)
        transcript.append(reply)
        calls = reply.get("tool_calls")
        if not calls:
            return transcript  # plain text: the final answer

        # Start every call of the step now. gather returns the outcomes in the order
        # of its arguments, which is the order of the calls, not of completion.
        outcomes = await asyncio.gather(*(execute(tools, call) for call in calls))
        for call, outcome in zip(calls, outcomes, strict=True):
            transcript.append({"role": "tool", "tool_call_id": call["id"], "content": outcome})

    return transcript  # the step limit: every call above has been answered


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
    if not isinstance(arguments, dict):  # valid JSON, but not an object of named arguments
        return error("invalid arguments")

    try:
        result = await tool(**arguments)
    except Exception as exc:  # noqa: BLE001  a tool may raise anything
        return error(f"tool failed: {exc}")
    return result if isinstance(result, str) else json.dumps(result)


def error(text: str) -> str:
    return json.dumps({"error": text})
