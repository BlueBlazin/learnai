# BUG: searches for special strings from the end of the text (the match that starts furthest right wins), so overlapping specials are resolved right to left instead of earliest position first.
from collections.abc import Callable


def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]:
    if allowed == "none_as_text":
        return encode_ordinary(text)

    by_length = sorted(specials, key=len, reverse=True)
    pieces: list[list[int]] = []  # collected back to front
    end = len(text)  # the unprocessed part is text[:end]
    pos = end - 1
    while pos >= 0:
        special = next(
            (s for s in by_length if pos + len(s) <= end and text.startswith(s, pos)), None
        )
        if special is None:
            pos -= 1
            continue
        if special not in allowed:
            raise ValueError(f"disallowed special token {special!r} at position {pos}")
        if pos + len(special) < end:
            pieces.append(encode_ordinary(text[pos + len(special) : end]))
        pieces.append([specials[special]])
        end = pos
        pos -= 1
    if end > 0:
        pieces.append(encode_ordinary(text[:end]))
    return [token for piece in reversed(pieces) for token in piece]
