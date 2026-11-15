import asyncio
import json


async def run_agent(model, tools: dict, messages: list[dict], max_steps: int) -> list[dict]:
    """Drive a chat model that calls tools until it answers or max_steps is reached.

    model: await model.acomplete(conversation) -> the assistant message (a dict);
        it has a "tool_calls" list only when the model requests tools.
    tools: {name: ScriptedTool}; await tool(**arguments) returns a result or raises.
    messages: the conversation so far; must not be modified.
    max_steps: the most model calls to make (at least 1).

    Returns the transcript: the given messages followed by every assistant message
    and, after each assistant message with tool calls, one
    {"role": "tool", "tool_call_id": ..., "content": ...} message per call, in call
    order.
    """
    raise NotImplementedError
