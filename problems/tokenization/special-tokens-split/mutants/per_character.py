# BUG: encodes ordinary text one character at a time instead of one call per run, so the tokenizer never sees two characters together and no merge is ever applied.
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
    ids: list[int] = []
    pos = 0
    while pos < len(text):
        special = next((s for s in by_length if text.startswith(s, pos)), None)
        if special is None:
            ids.extend(encode_ordinary(text[pos]))
            pos += 1
            continue
        if special not in allowed:
            raise ValueError(f"disallowed special token {special!r} at position {pos}")
        ids.append(specials[special])
        pos += len(special)
    return ids
