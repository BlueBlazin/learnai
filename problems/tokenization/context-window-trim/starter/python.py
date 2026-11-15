from collections.abc import Callable

Message = dict[str, str]


def trim_messages(
    messages: list[Message],
    count_tokens: Callable[[list[Message]], int],
    max_context: int,
    max_new_tokens: int,
) -> list[Message]:
    """Drop the oldest turns until the prompt and the reserved output fit.

    messages: the conversation, oldest first; dicts with "role" and "content". An
        optional system message comes first. A turn starts at each user message and
        runs up to the next user message.
    count_tokens: count_tokens(msgs) is the token count of the prompt rendered from
        the list msgs (not additive over messages).
    max_context: the context window in tokens.
    max_new_tokens: tokens reserved for the answer.
    Returns a new list: the system message (if any) and all turns except the fewest
    oldest ones such that count_tokens(result) + max_new_tokens <= max_context.
    Raises ValueError if the system message plus the last turn do not fit.
    """
    raise NotImplementedError
