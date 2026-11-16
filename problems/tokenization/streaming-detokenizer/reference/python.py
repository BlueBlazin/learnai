from typing import Any

from learnai.kit import ToyVocab


def detok_step(state: Any, token_id: int, vocab: ToyVocab) -> tuple[Any, str]:
    # The state is the bytes received but not yet emitted: at most three, the start of
    # one unfinished character. bytes are immutable, so old states stay valid.
    pending = (state or b"") + vocab.bytes(token_id)  # the only lookup: once per token
    n = _complete_prefix_len(pending)
    # Everything before the unfinished tail is final, including invalid bytes, which
    # decode to U+FFFD now exactly as they would in a decode of the whole stream.
    return pending[n:], pending[:n].decode("utf-8", errors="replace")


def detok_flush(state: Any) -> str:
    # Whatever is still held back will never be completed: it decodes to U+FFFD.
    return (state or b"").decode("utf-8", errors="replace")


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
