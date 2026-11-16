from learnai.kit import Chars, Grammar, Ref, ToyVocab


class GrammarMatcher:
    """Which tokens keep the output a prefix of a string of the grammar's language.

    grammar: a parsed grammar. grammar.rules maps each rule name to a tuple of
        alternatives; an alternative is a tuple of symbols; a symbol is Chars (one
        character of a set: chars.matches(code_point), chars.overlaps(low, high)) or
        Ref (ref.name is another rule). The language is what the rule "root" matches.
        grammar.nullable and grammar.left_recursive name the rules that can match
        nothing and the rules that can begin with themselves.
    vocab: the token table; vocab.bytes(token) is a token's byte string (b"" for a
        special token), vocab.size the number of tokens.
    States are opaque to the caller, hashable, and never modified.
    """

    def __init__(self, grammar: Grammar, vocab: ToyVocab) -> None:
        raise NotImplementedError

    def start(self):
        """The state before any token."""
        raise NotImplementedError

    def allowed(self, state) -> list[int]:
        """The ids, in increasing order, of the tokens with at least one byte whose
        bytes, appended to the text so far, leave a prefix of the UTF-8 encoding of
        some string of the language."""
        raise NotImplementedError

    def advance(self, state, token: int):
        """The state after `token`. Raises ValueError if the token is not allowed."""
        raise NotImplementedError

    def is_accepting(self, state) -> bool:
        """Whether the text so far, as it stands, is a string of the language."""
        raise NotImplementedError
