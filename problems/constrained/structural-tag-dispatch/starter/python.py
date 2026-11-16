from learnai.kit import DEAD, CharDFA, ToyVocab


class TagDispatch:
    """Free text until a trigger, then one of the trigger's tags, then free text again.

    triggers: non-empty strings. When the free text read since it began ends with a
        trigger, the trigger fires.
    tags: tuples (begin, inner, end). `begin` starts with a trigger; after the
        trigger fires the output must spell the begin of one of its tags, then
        content read by the CharDFA `inner` from its start state
        (inner.step(state, byte) -> next state or DEAD, inner.is_accepting(state)),
        then `end`, which may start only when `inner` is accepting and cannot read
        the byte.
    vocab: vocab.bytes(token) is a token's byte string (b"" for a special token),
        vocab.size the number of tokens.
    Raises ValueError if a trigger is a substring of another trigger, a begin starts
    with no trigger, or a begin is a prefix of another begin.
    States are opaque to the caller, hashable, and never modified.
    """

    def __init__(
        self, triggers: list[str], tags: list[tuple[str, CharDFA, str]], vocab: ToyVocab
    ) -> None:
        raise NotImplementedError

    def start(self):
        """The state before any token."""
        raise NotImplementedError

    def allowed(self, state) -> list[int]:
        """The ids, in increasing order, of the tokens with at least one byte whose
        bytes can all be read from this state."""
        raise NotImplementedError

    def advance(self, state, token: int):
        """The state after `token`. Raises ValueError if the token is not allowed."""
        raise NotImplementedError

    def is_accepting(self, state) -> bool:
        """Whether the output may end here: the reader is outside every tag."""
        raise NotImplementedError
