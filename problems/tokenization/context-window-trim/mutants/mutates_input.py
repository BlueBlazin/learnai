# BUG: trims the caller's list in place and returns it, so the application's copy of the history loses the dropped turns.
from collections.abc import Callable

Message = dict[str, str]


def trim_messages(
    messages: list[Message],
    count_tokens: Callable[[list[Message]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[Message]:
    first = 1 if messages[0]["role"] == "system" else 0
    budget = max_context - max_new_tokens
    while count_tokens(messages) > budget:
        starts = [i for i in range(first, len(messages)) if messages[i]["role"] == "user"]
        if len(starts) < 2:
            raise ValueError("the system message and the last turn do not fit")
        del messages[first : starts[1]]
    return messages
