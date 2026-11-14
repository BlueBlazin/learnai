# BUG: when several special strings start at the same position, takes the shortest instead of the longest, so a special that extends another one is never produced.
from collections.abc import Callable


def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]:
    if allowed == "none_as_text":
        return encode_ordinary(text)  # special strings are just characters

    by_length = sorted(specials, key=len)
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
    if run_start < len(text):
        ids.extend(encode_ordinary(text[run_start:]))
    return ids
