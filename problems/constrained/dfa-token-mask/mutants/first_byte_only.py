# BUG: allowed_tokens checks only the first byte of each token; a multi-byte token whose later bytes leave the language is still allowed.
from learnai.kit import DEAD, CharDFA, ToyVocab


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    allowed = []
    for token in range(vocab.size):
        piece = vocab.bytes(token)
        if piece and dfa.step(state, piece[0]) != DEAD:
            allowed.append(token)
    return allowed


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    piece = vocab.bytes(token)
    after = dfa.walk(state, piece) if piece else DEAD
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
