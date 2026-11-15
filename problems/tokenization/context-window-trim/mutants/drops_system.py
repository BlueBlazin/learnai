# BUG: treats the system message as part of the oldest turn, so it is dropped with it.
from collections.abc import Callable

Message = dict[str, str]


def trim_messages(
    messages: list[Message],
    count_tokens: Callable[[list[Message]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[Message]:
    turn_starts = [i for i, message in enumerate(messages) if message["role"] == "user"]
    budget = max_context - max_new_tokens
    if count_tokens(messages) <= budget:
        return list(messages)
    for start in turn_starts:
        kept = messages[start:]
        if count_tokens(kept) <= budget:
            return kept
    raise ValueError("the last turn does not fit in the context window")
