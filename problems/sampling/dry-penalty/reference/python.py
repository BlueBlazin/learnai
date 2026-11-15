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

    # A match is a run of tokens at the end of the window, and a breaker cannot be
    # part of one, so no match is longer than the run of non-breakers at the end.
    limit = 0
    while limit < n and window[n - 1 - limit] not in breakers:
        limit += 1
    if limit < allowed_length:
        return {}

    # Reverse the window: "the longest suffix of the window that also ends at
    # position i" becomes "the longest prefix of rev that also starts at k", with
    # k = n - 1 - i. That is the Z-array of rev, computed in linear time: [left, right)
    # is the rightmost stretch known to equal a prefix of rev, and inside it z[k] can
    # be copied from z[k - left] unless the copy reaches the end of the stretch.
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

    # rev[k - 1] is the token that followed the earlier occurrence: emitting it now
    # would extend the repeat. Keep the longest match per candidate token.
    longest: dict[int, int] = {}
    for k in range(1, n):
        length = min(z[k], limit)
        if length >= allowed_length:
            candidate = rev[k - 1]
            if candidate not in breakers and length > longest.get(candidate, 0):
                longest[candidate] = length

    # The exponent is clamped where base ** exponent would overflow a float32.
    max_exponent = int(FLOAT_MAX_LOG / math.log(base)) if base > 1.000001 else 0
    penalties = {}
    for candidate, length in longest.items():
        exponent = length - allowed_length
        if 0 < max_exponent < exponent:
            exponent = max_exponent
        penalties[candidate] = multiplier * base**exponent
    return penalties
