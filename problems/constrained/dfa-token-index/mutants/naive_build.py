# BUG: build_index walks every token separately from every state, repeating the steps of shared prefixes (the step for b't' is taken again for b'tr', b'tru', and b'true'), which exceeds the step budget.
from learnai.kit import DEAD, CharDFA, ToyVocab


def build_index(dfa: CharDFA, vocab: ToyVocab):
    pieces = [vocab.bytes(token) for token in range(vocab.size)]
    index: dict[int, dict[int, int]] = {}
    for state in dfa.states:
        moves: dict[int, int] = {}
        for token, piece in enumerate(pieces):
            after = dfa.walk(state, piece) if piece else DEAD
            if after != DEAD:
                moves[token] = after
        index[state] = moves
    return index


def allowed_from_index(index, state: int) -> list[int]:
    return sorted(index[state])


def advance_from_index(index, state: int, token: int) -> int:
    try:
        return index[state][token]
    except KeyError:
        raise ValueError(f"token {token} is not allowed in state {state}") from None
