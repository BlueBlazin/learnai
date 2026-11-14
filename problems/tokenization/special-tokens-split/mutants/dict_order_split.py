# BUG: handles the special tokens one at a time in dictionary order, splitting the text on the first special everywhere before looking for the next, so the order of `specials` decides instead of "earliest position, then longest".
from collections.abc import Callable


def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]:
    if allowed == "none_as_text":
        return encode_ordinary(text)

    parts: list[str | int] = [text]  # strings are still text, ints are finished ids
    for special, token_id in specials.items():
        split: list[str | int] = []
        for part in parts:
            if isinstance(part, int) or special not in part:
                split.append(part)
                continue
            if special not in allowed:
                raise ValueError(f"disallowed special token {special!r}")
            chunks = part.split(special)
            for i, chunk in enumerate(chunks):
                if i > 0:
                    split.append(token_id)
                if chunk:
                    split.append(chunk)
        parts = split
    ids: list[int] = []
    for part in parts:
        if isinstance(part, int):
            ids.append(part)
        else:
            ids.extend(encode_ordinary(part))
    return ids
