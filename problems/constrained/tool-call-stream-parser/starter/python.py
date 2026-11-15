import json

OPEN = "<tool_call>"
CLOSE = "</tool_call>"


def parse_start():
    """The parser state before any text has arrived. Any value; never modified later."""
    raise NotImplementedError


def parse_step(state, piece: str):
    """Consume one piece of the stream.

    state: a state returned by parse_start or parse_step. It must not be modified.
    piece: the next characters of the stream (possibly none).
    Returns (new_state, events). events is a list of tuples, in text order:
        ("content", text)                    text outside tool calls
        ("tool_call_start", index, name)     at the closing quote of the name
        ("arguments", index, text)           the next characters of the arguments object
        ("tool_call_end", index)             at the final ">" of </tool_call>
    Content is emitted as it arrives, except that the longest suffix that is a proper
    prefix of "<tool_call>" is held back. Adjacent content events (and adjacent
    arguments events of one call) within a step are merged; no text is empty.
    """
    raise NotImplementedError


def parse_end(state):
    """The stream is over. Returns [("content", held)] for held-back content, else []."""
    raise NotImplementedError
