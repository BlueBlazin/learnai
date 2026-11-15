# BUG: takes the forbidden shortcut: the state is the text so far, and every question is passed to the grammar's test oracle instead of being answered by a matcher.
from whetstone.kit import Grammar, ToyVocab


class GrammarMatcher:
    def __init__(self, grammar: Grammar, vocab: ToyVocab) -> None:
        self.grammar = grammar
        self.pieces = [vocab.bytes(token) for token in range(vocab.size)]

    def start(self):
        return b""

    def allowed(self, state) -> list[int]:
        return [
            token
            for token, piece in enumerate(self.pieces)
            if piece and self.grammar.accepts_prefix(state + piece)
        ]

    def advance(self, state, token: int):
        piece = self.pieces[token]
        if not piece or not self.grammar.accepts_prefix(state + piece):
            raise ValueError(f"token {token} is not allowed in this state")
        return state + piece

    def is_accepting(self, state) -> bool:
        return self.grammar.accepts(state)
