# BUG: drops the most recent turns before the last one first and keeps the oldest history, instead of dropping the oldest turns.
from collections.abc import Callable

Message = dict[str, str]


def trim_messages(
    messages: list[Message],
    count_tokens: Callable[[list[Message]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[Message]:
    head = messages[:1] if messages[0]["role"] == "system" else []
    body = messages[len(head) :]
    turn_starts = [i for i, message in enumerate(body) if message["role"] == "user"]
    last = body[turn_starts[-1] :]
    budget = max_context - max_new_tokens
    for end in reversed(turn_starts):
        kept = head + body[:end] + last
        if count_tokens(kept) <= budget:
            return kept
    raise ValueError("the system message and the last turn do not fit in the context window")
