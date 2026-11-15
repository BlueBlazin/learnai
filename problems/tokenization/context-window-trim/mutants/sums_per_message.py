# BUG: counts each message on its own and adds the counts, although the count of a prompt is not the sum over its messages.
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
    cost = [count_tokens([message]) for message in body]
    head_cost = sum(count_tokens([message]) for message in head)
    for start in turn_starts:
        if head_cost + sum(cost[start:]) <= budget:
            return head + body[start:]
    raise ValueError("the system message and the last turn do not fit in the context window")
