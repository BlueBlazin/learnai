# BUG: compares the prompt with max_context alone and reserves nothing for the answer.
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
    for start in turn_starts:
        kept = head + body[start:]
        if count_tokens(kept) <= max_context:
            return kept
    raise ValueError("the system message and the last turn do not fit in the context window")
