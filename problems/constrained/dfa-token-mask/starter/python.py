from whetstone.kit import DEAD, CharDFA, ToyVocab


def allowed_tokens(dfa: CharDFA, state: int, vocab: ToyVocab) -> list[int]:
    """The tokens that may be generated next when the DFA is in `state`.

    dfa: the automaton of the language; read it with dfa.step(state, byte) or
        dfa.walk(state, data), which return the next state or DEAD.
    state: a state of the DFA (never DEAD).
    vocab: the token table; vocab.bytes(token) is the byte string of a token and
        vocab.size the number of tokens.
    Returns the ids, in increasing order, of every token that has at least one byte
    and whose bytes can all be read from `state` without the walk becoming DEAD. The
    walk may end in any state, accepting or not.
    """
    raise NotImplementedError


def advance(dfa: CharDFA, state: int, token: int, vocab: ToyVocab) -> int:
    """The DFA state after generating `token` in `state`.

    Returns the state reached by reading the token's bytes from `state`.
    Raises ValueError if the token is not allowed in `state` (see allowed_tokens).
    """
    raise NotImplementedError
