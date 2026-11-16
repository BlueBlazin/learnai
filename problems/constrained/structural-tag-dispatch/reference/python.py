from learnai.kit import DEAD, CharDFA, ToyVocab


class TagDispatch:
    """Free text until a trigger, then one of the tags that begin with it, then free text.

    A state is a tuple:
        ("free", held)              outside tags; `held` is the longest suffix of the
                                    free text that is a proper prefix of a trigger
        ("begin", candidates, n)    after a trigger; n bytes of the begin strings of
                                    the tags `candidates` have been matched
        ("inner", tag, q)           inside the tag's content, DFA state q
        ("end", tag, n)             n bytes of the tag's end string have been matched
    """

    def __init__(
        self, triggers: list[str], tags: list[tuple[str, CharDFA, str]], vocab: ToyVocab
    ) -> None:
        self.triggers = [trigger.encode() for trigger in triggers]
        self.tags = [(begin.encode(), inner, end.encode()) for begin, inner, end in tags]
        for i, trigger in enumerate(self.triggers):
            if not trigger:
                raise ValueError("a trigger is empty")
            if any(trigger in other for j, other in enumerate(self.triggers) if j != i):
                raise ValueError(f"trigger {trigger!r} is contained in another trigger")
        for i, (begin, _, _) in enumerate(self.tags):
            if not any(begin.startswith(trigger) for trigger in self.triggers):
                raise ValueError(f"begin {begin!r} starts with no trigger")
            if any(other.startswith(begin) for j, (other, _, _) in enumerate(self.tags) if j != i):
                raise ValueError(f"begin {begin!r} is a prefix of another begin")
        self.pieces = [vocab.bytes(token) for token in range(vocab.size)]

    def start(self):
        return ("free", b"")

    def allowed(self, state) -> list[int]:
        # A special token has no bytes: a walk over nothing would always survive.
        return [
            token
            for token, piece in enumerate(self.pieces)
            if piece and self._walk(state, piece) is not None
        ]

    def advance(self, state, token: int):
        piece = self.pieces[token]
        after = self._walk(state, piece) if piece else None
        if after is None:
            raise ValueError(f"token {token} is not allowed in this state")
        return after

    def is_accepting(self, state) -> bool:
        return state[0] == "free"

    # -- the byte-level automaton ----------------------------------------------------------

    def _walk(self, state, piece: bytes):
        """The state after all bytes of `piece`, or None if one of them cannot follow.

        A token is not a unit of structure: its bytes may finish a trigger, run
        through a begin string, and reach into the content.
        """
        for byte in piece:
            state = self._step(state, byte)
            if state is None:
                return None
        return state

    def _step(self, state, byte: int):
        kind = state[0]
        if kind == "free":
            held = state[1] + bytes([byte])
            # The longest suffix that can still grow into a trigger. Dropping one
            # byte at a time from the front is what finds "<<fn=" inside "<<<fn=".
            while held and not any(trigger.startswith(held) for trigger in self.triggers):
                held = held[1:]
            if held not in self.triggers:
                return ("free", held)
            candidates = tuple(
                i for i, (begin, _, _) in enumerate(self.tags) if begin.startswith(held)
            )
            return self._begun(candidates, len(held))
        if kind == "begin":
            _, candidates, matched = state
            candidates = tuple(i for i in candidates if self.tags[i][0][matched] == byte)
            return self._begun(candidates, matched + 1)
        if kind == "inner":
            _, tag, q = state
            _, inner, end = self.tags[tag]
            after = inner.step(q, byte)
            if after != DEAD:
                return ("inner", tag, after)
            # The content cannot go on with this byte. It may stop here if it is
            # complete and the byte starts the end string.
            if inner.is_accepting(q) and byte == end[0]:
                return self._ended(tag, 1)
            return None
        _, tag, matched = state
        if byte == self.tags[tag][2][matched]:
            return self._ended(tag, matched + 1)
        return None

    def _begun(self, candidates: tuple[int, ...], matched: int):
        """The state after `matched` bytes of the begin strings of `candidates`."""
        if not candidates:
            return None  # the text follows no tag (or the trigger has none)
        first = candidates[0]
        if len(self.tags[first][0]) == matched:
            # A begin string is complete. No begin is a prefix of another, so it is
            # the only candidate left; its content starts from the DFA's start state,
            # every time.
            return ("inner", first, self.tags[first][1].start)
        return ("begin", candidates, matched)

    def _ended(self, tag: int, matched: int):
        if matched == len(self.tags[tag][2]):
            return ("free", b"")  # the tag is closed; scanning starts afresh
        return ("end", tag, matched)
