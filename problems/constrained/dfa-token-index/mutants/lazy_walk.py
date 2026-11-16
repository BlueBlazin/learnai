# BUG: build_index only stores the DFA and the vocabulary; every query walks the tokens through the DFA again, so nothing is precomputed and each mask costs DFA steps.
from learnai.kit import DEAD, CharDFA, ToyVocab


def build_index(dfa: CharDFA, vocab: ToyVocab):
    return dfa, [vocab.bytes(token) for token in range(vocab.size)]


def _walk(dfa: CharDFA, state: int, piece: bytes) -> int:
    return dfa.walk(state, piece) if piece else DEAD


def allowed_from_index(index, state: int) -> list[int]:
    dfa, pieces = index
    return [token for token, piece in enumerate(pieces) if _walk(dfa, state, piece) != DEAD]


def advance_from_index(index, state: int, token: int) -> int:
    dfa, pieces = index
    after = _walk(dfa, state, pieces[token])
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
