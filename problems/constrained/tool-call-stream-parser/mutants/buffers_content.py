# BUG: content is emitted only when a tool call begins or the stream ends; the merged result is right, but text that is certain to be content is withheld for many steps instead of being streamed.
import json
from typing import NamedTuple

OPEN = "<tool_call>"
CLOSE = "</tool_call>"


class State(NamedTuple):
    """Everything the parser remembers between pieces. Immutable."""

    mode: str = "content"  # "content", "head", "arguments", or "tail"
    index: int = 0  # the index of the next tool call (or of the open one)
    held: str = ""  # content mode: text that may be the beginning of OPEN
    #                 head mode: the string literal being read
    #                 tail mode: the text after the arguments
    strings: int = 0  # head mode: string literals completed ("name", the name, "arguments")
    depth: int = 0  # arguments mode: open braces and brackets
    in_string: bool = False
    escaped: bool = False  # the previous character was a backslash inside a string


def parse_start() -> State:
    return State()


def _emit(events: list, kind: str, *fields) -> None:
    """Append a delta, merging it into the previous event when that is the same kind."""
    *key, text = fields
    if events and events[-1][0] == kind and list(events[-1][1:-1]) == key:
        events[-1] = (kind, *key, events[-1][-1] + text)
    else:
        events.append((kind, *key, text))


def parse_step(state: State, piece: str) -> tuple[State, list[tuple]]:
    mode, index, held, strings, depth, in_string, escaped = state
    events: list[tuple] = []
    for ch in piece:
        if mode == "content":
            held += ch
            if held.endswith(OPEN):
                if len(held) > len(OPEN):
                    _emit(events, "content", held[: -len(OPEN)])
                mode, held, strings = "head", "", 0
        elif mode == "head":
            # {"name": "<name>", "arguments": <object>: three string literals, then
            # the brace that opens the arguments.
            if in_string:
                held += ch
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                    strings += 1
                    if strings == 2:  # the name is complete, quotes and escapes included
                        events.append(("tool_call_start", index, json.loads(held)))
            elif ch == '"':
                in_string, held = True, ch
            elif ch == "{" and strings == 3:
                mode, depth, held = "arguments", 1, ""
                _emit(events, "arguments", index, ch)
        elif mode == "arguments":
            _emit(events, "arguments", index, ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    mode, held = "tail", ""
        else:  # tail: whitespace, the brace that closes the call object, the end tag
            held += ch
            if held.endswith(CLOSE):
                events.append(("tool_call_end", index))
                mode, index, held = "content", index + 1, ""
    return State(mode, index, held, strings, depth, in_string, escaped), events


def parse_end(state: State) -> list[tuple]:
    # Everything since the last tool call.
    if state.mode == "content" and state.held:
        return [("content", state.held)]
    return []
