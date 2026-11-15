# BUG: returns the system message and the last turn even when they do not fit, instead of raising ValueError.
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
    budget = max_context - max_new_tokens
    for start in turn_starts:
        kept = head + body[start:]
        if count_tokens(kept) <= budget:
            return kept
    return head + body[turn_starts[-1] :]
