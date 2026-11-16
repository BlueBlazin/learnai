# BUG: tokens are recorded only at the leaves of the vocabulary trie, so a token whose bytes are a prefix of another token (b't' under b'tr', b'tr' under b'true') is missing from the index.
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
    index: dict[int, dict[int, int]] = {}
    for state in dfa.states:
        moves: dict[int, int] = {}
        # Walk the trie and the DFA together. Each trie edge is stepped once for this
        # state, so the step for "t" is shared by "tr", "tru", and "true".
        stack = [(root, state)]
        while stack:
            node, at = stack.pop()
            for byte, child in node.children.items():
                after = dfa.step(at, byte)
                if after == DEAD:
                    continue  # every token below this node dies here: skip the subtree
                if not child.children:  # the end of a branch
                    for token in child.tokens:
                        moves[token] = after
                stack.append((child, after))
        index[state] = moves
    return index


def allowed_from_index(index, state: int) -> list[int]:
    return sorted(index[state])


def advance_from_index(index, state: int, token: int) -> int:
    try:
        return index[state][token]
    except KeyError:
        raise ValueError(f"token {token} is not allowed in state {state}") from None
