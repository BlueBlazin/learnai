# BUG: assumes strict user/assistant alternation and drops two messages at a time, which cuts through turns that contain tool messages.
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
    last_turn = max(i for i, message in enumerate(body) if message["role"] == "user")
    budget = max_context - max_new_tokens
    start = 0
    while True:
        kept = head + body[start:]
        if count_tokens(kept) <= budget:
            return kept
        if start >= last_turn:
            raise ValueError("the system message and the last turn do not fit")
        start = min(start + 2, last_turn)
