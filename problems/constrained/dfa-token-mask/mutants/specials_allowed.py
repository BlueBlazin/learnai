# BUG: special tokens are not excluded; their byte string is empty, a walk over zero bytes never dies, and so they are allowed in every state.
from whetstone.kit import DEAD, CharDFA, ToyVocab


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    return [t for t in range(vocab.size) if dfa.walk(state, vocab.bytes(t)) != DEAD]


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    after = dfa.walk(state, vocab.bytes(token))
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
