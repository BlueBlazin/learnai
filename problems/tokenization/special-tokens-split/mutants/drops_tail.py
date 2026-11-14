# BUG: forgets the ordinary text after the last special token (and the whole text when it contains no special).
from collections.abc import Callable


def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]:
    if allowed == "none_as_text":
        return encode_ordinary(text)  # special strings are just characters

    # Longest first, so the first hit at a position is the longest special there.
    by_length = sorted(specials, key=len, reverse=True)
    ids: list[int] = []
    run_start = 0  # where the current run of ordinary text began
    pos = 0
    while pos < len(text):
        special = next((s for s in by_length if text.startswith(s, pos)), None)
        if special is None:
            pos += 1
            continue
        if special not in allowed:
            # Refuse rather than guess: the caller did not expect control tokens here.
            raise ValueError(f"disallowed special token {special!r} at position {pos}")
        if pos > run_start:
            ids.extend(encode_ordinary(text[run_start:pos]))  # one call per run of text
        ids.append(specials[special])
        pos += len(special)  # matches never overlap: resume after the special
        run_start = pos
    return ids
