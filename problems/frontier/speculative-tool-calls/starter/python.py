import asyncio
import json

from learnai.kit import assemble


async def run_agent(model, tools: dict, messages: list[dict], predictor, max_steps: int) -> list:
    """A tool-calling loop that starts safe tool calls before the model's turn is over.

    model: `model.stream(transcript)` is an async iterator over the pieces of one turn;
        `assemble(pieces)` turns the pieces of a turn into the assistant message.
    tools: {name: ScriptedTool}. `await tool(**arguments)` runs it; `tool.side_effects`
        is True for a tool that must not be started before the turn has ended.
    messages: the conversation so far (not modified).
    predictor: `predictor(name, arguments_so_far)` returns the arguments it expects the
        call to end up with (a dict), or None.
    max_steps: the most model turns to run.
    Returns the transcript: the messages, then every assistant and tool message appended.
    """
    raise NotImplementedError
