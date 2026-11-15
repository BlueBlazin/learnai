"""TokenConstraint: what a decoding loop needs to know to stay inside a language.

A constrained decoder does not care whether the rule is a regex, a JSON schema, or a
list of choices. It needs four operations, and this module names them:

    state = constraint.start()
    while ...:
        allowed = constraint.allowed(state)          # token ids that may come next
        token = ...pick one of them...
        state = constraint.advance(state, token)
    constraint.is_accepting(state)                   # may the output end here?

States are opaque, hashable, and never modified: ``advance`` returns a new state, so a
loop can keep an old one to branch or roll back.

``DFAConstraint`` is the reference implementation, for a regular language given as a
``CharDFA`` over a ``ToyVocab``.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Protocol, runtime_checkable

from whetstone.kit.counters import Counters
from whetstone.kit.dfa import DEAD, CharDFA
from whetstone.kit.vocab import ToyVocab


@runtime_checkable
class TokenConstraint(Protocol):
    """The interface between a decoding loop and a constraint."""

    def start(self) -> Hashable:
        """The state before any token has been generated."""
        ...

    def allowed(self, state: Hashable) -> list[int]:
        """The token ids that may be generated next, in increasing order.

        End-of-sequence is never in the list: whether the output may stop is
        ``is_accepting``. An empty list means no token can follow.
        """
        ...

    def advance(self, state: Hashable, token: int) -> Hashable:
        """The state after generating ``token``; ``ValueError`` if it was not allowed."""
        ...

    def is_accepting(self, state: Hashable) -> bool:
        """Whether the tokens generated so far form a complete output."""
        ...


class DFAConstraint:
    """Tokens of ``vocab`` whose text keeps the output inside the language of ``dfa``.

    The state is the DFA state reached by the bytes of all tokens so far. A token is
    allowed when it has at least one byte and walking its bytes from the state ends
    where an accepting state is still reachable. Special tokens have no bytes, so they
    are never allowed.

    The answers for each state are computed once, on first use, without touching
    ``dfa.counters``. ``constraint.counters`` counts ``allowed_calls`` and
    ``advance_calls``.
    """

    def __init__(self, dfa: CharDFA, vocab: ToyVocab) -> None:
        self.dfa = dfa
        self.vocab = vocab
        self.counters = Counters("allowed_calls", "advance_calls")
        self._index: dict[int, dict[int, int]] = {}

    def with_framework(self, fw: Any) -> DFAConstraint:
        """The same constraint with fresh counters."""
        twin = object.__new__(DFAConstraint)
        twin.__dict__.update(self.__dict__)
        twin.counters = Counters("allowed_calls", "advance_calls")
        return twin

    def start(self) -> int:
        return self.dfa.start

    def allowed(self, state: int) -> list[int]:
        self.counters.add("allowed_calls")
        return sorted(self._moves(state))

    def advance(self, state: int, token: int) -> int:
        self.counters.add("advance_calls")
        moves = self._moves(state)
        try:
            return moves[int(token)]
        except KeyError, TypeError, ValueError:
            raise ValueError(f"token {token!r} is not allowed in state {state!r}") from None

    def is_accepting(self, state: int) -> bool:
        return self.dfa.is_accepting(state)

    def _moves(self, state: int) -> dict[int, int]:
        """``{token id: next state}`` for every token allowed in ``state``."""
        if state not in self._index:
            moves = {}
            for token, piece in enumerate(self.vocab.pieces):
                after = state if piece else DEAD
                for byte in piece:
                    after = self.dfa._next(after, byte)
                if self.dfa.is_live(after):
                    moves[token] = after
            self._index[state] = moves
        return self._index[state]
