# BUG: checks only whether the last token is a breaker; a breaker further back does not stop a match from extending through it.
import math

FLOAT_MAX_LOG = 88.7228391  # ln of the largest float32


def dry_penalties(
    history: list[int],
    multiplier: float,
    base: float,
    allowed_length: int,
    breakers: set[int],
    last_n: int | None = None,
) -> dict[int, float]:
    window = history if last_n is None else history[-last_n:]
    n = len(window)

    limit = 0 if n and window[-1] in breakers else n
    if limit < allowed_length:
        return {}

    rev = window[::-1]
    z = [0] * n
    left = right = 0
    for k in range(1, n):
        if k < right and z[k - left] < right - k:
            z[k] = z[k - left]
            continue
        length = max(0, right - k)
        while k + length < n and rev[length] == rev[k + length]:
            length += 1
        z[k] = length
        left, right = k, k + length

    longest: dict[int, int] = {}
    for k in range(1, n):
        length = min(z[k], limit)
        if length >= allowed_length:
            candidate = rev[k - 1]
            if candidate not in breakers and length > longest.get(candidate, 0):
                longest[candidate] = length

    max_exponent = int(FLOAT_MAX_LOG / math.log(base)) if base > 1.000001 else 0
    penalties = {}
    for candidate, length in longest.items():
        exponent = length - allowed_length
        if 0 < max_exponent < exponent:
            exponent = max_exponent
        penalties[candidate] = multiplier * base**exponent
    return penalties
