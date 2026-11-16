# BUG: tokens whose bytes are not valid UTF-8 on their own (a piece of a multi-byte character) are treated as unusable and never allowed, so characters split across tokens cannot be generated.
from learnai.kit import DEAD, CharDFA, ToyVocab


def _walk_token(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    piece = vocab.bytes(token)
    if not piece:
        return DEAD
    try:
        piece.decode("utf-8")
    except UnicodeDecodeError:
        return DEAD
    return dfa.walk(state, piece)


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    return [t for t in range(vocab.size) if _walk_token(dfa, state, t, vocab) != DEAD]


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    after = _walk_token(dfa, state, token, vocab)
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
