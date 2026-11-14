# BUG: the state is a bytearray that detok_step extends and trims in place, so the state object the caller still holds changes under it and continuing twice from one state goes wrong.
from typing import Any

from whetstone.kit import ToyVocab


def detok_step(state: Any, token_id: int, vocab: ToyVocab) -> tuple[Any, str]:
    pending = bytearray() if state is None else state
    pending += vocab.bytes(token_id)  # in place: the caller's state object changes
    n = _complete_prefix_len(pending)
    text = bytes(pending[:n]).decode("utf-8", errors="replace")
    del pending[:n]
    return pending, text


def detok_flush(state: Any) -> str:
    return bytes(state or b"").decode("utf-8", errors="replace")


def _complete_prefix_len(data: bytes) -> int:
    """Length of `data` without an unfinished, still valid UTF-8 character at its end.

    This is the function of the `utf8-boundaries` problem.
    """
    n = len(data)
    for back in range(1, min(3, n) + 1):
        byte = data[n - back]
        if byte & 0xC0 == 0x80:
            continue  # a continuation byte: keep looking for its lead byte
        if 0xC2 <= byte <= 0xDF:
            need = 2
        elif 0xE0 <= byte <= 0xEF:
            need = 3
        elif 0xF0 <= byte <= 0xF4:
            need = 4
        else:
            return n  # ASCII or a byte that starts nothing: complete
        if back >= need:
            return n  # the character has all its bytes
        if back >= 2:
            second = data[n - back + 1]
            low = {0xE0: 0xA0, 0xF0: 0x90}.get(byte, 0x80)
            high = {0xED: 0x9F, 0xF4: 0x8F}.get(byte, 0xBF)
            if not low <= second <= high:
                return n  # already invalid: no byte can complete it
        return n - back
    return n
