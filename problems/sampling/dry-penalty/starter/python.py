import math

FLOAT_MAX_LOG = 88.7228391  # ln of the largest float32, for the exponent clamp


def dry_penalties(
    history: list[int],
    multiplier: float,
    base: float,
    allowed_length: int,
    breakers: set[int],
    last_n: int | None = None,
) -> dict[int, float]:
    """The DRY penalty of every token that would extend a repeat of the history.

    history: the token ids generated so far (prompt included), oldest first.
    multiplier > 0, base >= 1, allowed_length >= 1: the penalty parameters.
    breakers: token ids that no match may contain and that are never penalized.
    last_n: if not None, only the last `last_n` >= 1 tokens of the history count.
    Returns a dict that maps each penalized token to the amount to subtract from its
    logit, multiplier * base ** (match length - allowed_length), with the exponent
    clamped as the statement describes. Tokens without a penalty are absent.
    """
    raise NotImplementedError
