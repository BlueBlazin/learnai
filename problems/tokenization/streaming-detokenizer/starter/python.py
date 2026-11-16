from typing import Any

from learnai.kit import ToyVocab


def detok_step(state: Any, token_id: int, vocab: ToyVocab) -> tuple[Any, str]:
    """Feed one generated token; return the new state and the text that is now final.

    state: None for the first token of a stream, otherwise the state returned by the
        previous call for this stream. Its type is yours to choose. Never modify a
        state you were given: the caller may continue from it again.
    token_id: the next token of the stream.
    vocab: the token table. vocab.bytes(token_id) is the token's bytes (b"" for a
        special token); every byte it hands out is counted.
    Returns (new_state, text). `text` is the newly final text: everything decodable
    from the bytes so far that was not returned by an earlier step, holding back only
    an unfinished UTF-8 character at the very end. Often "".
    """
    raise NotImplementedError


def detok_flush(state: Any) -> str:
    """End of the stream: return the text for whatever `state` still holds back.

    state: None (no token was fed) or a state returned by detok_step.
    Returns "" if nothing is held back, otherwise the held-back bytes decoded with
    errors="replace" (an unfinished character becomes one U+FFFD).
    """
    raise NotImplementedError
