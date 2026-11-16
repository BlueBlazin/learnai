# BUG: the index is keyed by token only: it records where each token leads from the start state and answers every query from that, whatever the current state is.
from learnai.kit import DEAD, CharDFA, ToyVocab


class _Node:
    """A node of the vocabulary trie: one distinct prefix of the tokens' byte strings."""

    __slots__ = ("children", "tokens")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}  # next byte -> node
        self.tokens: list[int] = []  # tokens whose bytes end exactly here


def _vocab_trie(vocab: ToyVocab) -> _Node:
    root = _Node()
    for token in range(vocab.size):
        piece = vocab.bytes(token)
        if not piece:
            continue  # special tokens have no text and are never allowed
        node = root
        for byte in piece:
            node = node.children.setdefault(byte, _Node())
        node.tokens.append(token)
    return root


def build_index(dfa: CharDFA, vocab: ToyVocab):
    root = _vocab_trie(vocab)
    moves: dict[int, int] = {}
    stack = [(root, dfa.start)]
    while stack:
        node, at = stack.pop()
        for byte, child in node.children.items():
            after = dfa.step(at, byte)
            if after == DEAD:
                continue
            for token in child.tokens:
                moves[token] = after
            stack.append((child, after))
    return moves


def allowed_from_index(index, state: int) -> list[int]:
    return sorted(index)


def advance_from_index(index, state: int, token: int) -> int:
    try:
        return index[token]
    except KeyError:
        raise ValueError(f"token {token} is not allowed in state {state}") from None
