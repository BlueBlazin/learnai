# BUG: decodes each token's bytes on its own, so a character whose bytes are spread over several tokens comes out as U+FFFD replacement characters.
from typing import Any

from whetstone.kit import ToyVocab


def detok_step(state: Any, token_id: int, vocab: ToyVocab) -> tuple[Any, str]:
    return None, vocab.bytes(token_id).decode("utf-8", errors="replace")


def detok_flush(state: Any) -> str:
    return ""
