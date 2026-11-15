# BUG: drops one message at a time instead of whole turns, so the result can start with an assistant or tool message whose user message is gone.
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
    for start in range(last_turn + 1):
        kept = head + body[start:]
        if count_tokens(kept) <= budget:
            return kept
    raise ValueError("the system message and the last turn do not fit in the context window")
