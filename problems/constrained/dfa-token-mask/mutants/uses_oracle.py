# BUG: takes the forbidden shortcut: decides with the DFA's test oracle `is_live` instead of comparing the walk's result with DEAD.
from learnai.kit import DEAD, CharDFA, ToyVocab


def _walk_token(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    piece = vocab.bytes(token)
    if not piece:
        return DEAD
    after = dfa.walk(state, piece)
    return after if dfa.is_live(after) else DEAD


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    return [t for t in range(vocab.size) if _walk_token(dfa, state, t, vocab) != DEAD]


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    after = _walk_token(dfa, state, token, vocab)
    if after == DEAD:
        raise ValueError(f"token {token} is not allowed in state {state}")
    return after
