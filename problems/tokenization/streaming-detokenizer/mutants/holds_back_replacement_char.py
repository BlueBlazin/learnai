# BUG: decides by looking at the decoded text: while it ends in U+FFFD, everything is held back. That delays invalid bytes, and a genuine U+FFFD character, until a later token (or the flush) releases them.
from typing import Any

from whetstone.kit import ToyVocab


def detok_step(state: Any, token_id: int, vocab: ToyVocab) -> tuple[Any, str]:
    pending = (state or b"") + vocab.bytes(token_id)
    text = pending.decode("utf-8", errors="replace")
    if text.endswith("\ufffd"):
        return pending, ""  # "probably" an unfinished character: wait
    return b"", text


def detok_flush(state: Any) -> str:
    return (state or b"").decode("utf-8", errors="replace")
