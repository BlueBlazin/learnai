# BUG: advance returns DEAD for a token that is not allowed instead of raising ValueError, so a decoding loop that picked a bad token carries on from a dead state.
from whetstone.kit import DEAD, CharDFA, ToyVocab


def _walk_token(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    piece = vocab.bytes(token)
    if not piece:
        return DEAD
    return dfa.walk(state, piece)


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    return [t for t in range(vocab.size) if _walk_token(dfa, state, t, vocab) != DEAD]


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    return _walk_token(dfa, state, token, vocab)
