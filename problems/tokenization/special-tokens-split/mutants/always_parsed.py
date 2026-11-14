# BUG: ignores `allowed`: every special string in the text becomes its id, also when it is not allowed and also in "none_as_text" mode. This is the prompt-injection hole.
from collections.abc import Callable


def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]:
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
        if pos > run_start:
            ids.extend(encode_ordinary(text[run_start:pos]))  # one call per run of text
        ids.append(specials[special])
        pos += len(special)  # matches never overlap: resume after the special
        run_start = pos
    if run_start < len(text):
        ids.extend(encode_ordinary(text[run_start:]))
    return ids
