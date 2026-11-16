from learnai.kit import DEAD, CharDFA, ToyVocab


def build_index(dfa: CharDFA, vocab: ToyVocab):
    """Precompute, for every state of the DFA, which tokens are allowed and where they lead.

    dfa: the automaton; dfa.states lists its states, dfa.step(state, byte) returns the
        next state or DEAD and is counted.
    vocab: the token table; vocab.bytes(token) for token in range(vocab.size).
    Returns the index, in any representation you like (the natural one is
    {state: {token: next_state}}): the tests only pass it back to the two functions
    below. Must stay within the step budget of the statement: one dfa.step per
    (state, distinct token prefix), and none below a prefix that already died.
    """
    raise NotImplementedError


def allowed_from_index(index, state: int) -> list[int]:
    """The ids, in increasing order, of the tokens allowed in `state`.

    Same answer as `allowed_tokens` of dfa-token-mask, without stepping the DFA.
    """
    raise NotImplementedError


def advance_from_index(index, state: int, token: int) -> int:
    """The state after generating `token` in `state`, without stepping the DFA.

    Raises ValueError if the token is not allowed in `state`.
    """
    raise NotImplementedError
