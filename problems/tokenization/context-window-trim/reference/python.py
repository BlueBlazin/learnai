from collections.abc import Callable

Message = dict[str, str]


def trim_messages(
    messages: list[Message],
    count_tokens: Callable[[list[Message]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[Message]:
    # The system message, if any, is kept whatever happens.
    head = messages[:1] if messages[0]["role"] == "system" else []
    body = messages[len(head) :]
    # Every user message opens a turn; the turn runs up to the next user message.
    turn_starts = [i for i, message in enumerate(body) if message["role"] == "user"]
    budget = max_context - max_new_tokens  # what is left for the prompt
    # Drop no turn, then the oldest, then the two oldest, ...: the first candidate
    # that fits keeps as much history as possible. The last candidate is the system
    # message plus the last turn.
    for start in turn_starts:
        kept = head + body[start:]
        if count_tokens(kept) <= budget:
            return kept
    raise ValueError("the system message and the last turn do not fit in the context window")
