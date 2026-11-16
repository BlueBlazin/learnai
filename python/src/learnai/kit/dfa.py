"""CharDFA: a deterministic finite automaton over bytes, for constrained generation.

A constrained decoder asks, before every token, "which tokens keep the output inside
the language?". When the language is regular, the answer comes from a DFA: feed the
token's bytes through it and see whether it survives.

    dfa = CharDFA.from_regex(r"-?[0-9]+")
    state = dfa.start
    state = dfa.step(state, ord("-"))      # a new state, or DEAD
    state = dfa.walk(state, b"12")         # step through several bytes
    dfa.is_accepting(state)                # True: "-12" is in the language

States are ints, the start state of a DFA built from a regex is 0, and ``DEAD`` (-1)
means "no string with this prefix is in the language". Stepping from ``DEAD`` stays
``DEAD``. The automaton reads **bytes** (ints 0 to 255), not characters, because
tokens are byte strings and may end in the middle of a UTF-8 character.

Every ``step`` is counted in ``dfa.counters.steps`` (``walk`` counts one step per byte
it consumes). The transition table itself is private: ``step`` is how a solution
learns about it, which is what lets the cost judge measure the work.

A DFA built by ``from_regex`` is minimal and *trimmed*: it has no state from which
acceptance is unreachable, so "the walk did not die" means exactly "this is a prefix
of some string in the language". A hand-built DFA may contain such trap states;
``trim()`` removes them.

Regex subset accepted by ``from_regex`` (the whole string must match):

    abc        literal characters; non-ASCII characters match their UTF-8 bytes
    .          any character except a newline
    [abc] [a-z0-9_] [^"]     classes; ranges must be ASCII; ^ negates
    \\d \\w \\s      ASCII digits, word characters, whitespace (also inside classes)
    \\D \\W \\S      their complements (outside classes only)
    \\n \\t \\r \\f \\v \\0 \\xHH     control characters and raw bytes
    \\. \\{ \\" ...   any escaped punctuation is literal
    ab  a|b  (ab)  (?:ab)     concatenation, alternation, grouping
    a*  a+  a?  a{3}  a{2,}  a{2,5}     repetition (bounds up to 100)

``.``, negated classes, and ``\\D \\W \\S`` match one whole character, ASCII or a
multi-byte UTF-8 sequence, never half of one. Not supported, and rejected with
``ValueError``: anchors (``^ $``), lazy and possessive quantifiers, backreferences,
lookaround, named groups, flags, Unicode classes, and non-ASCII range endpoints. The
metacharacters ``. ^ $ * + ? { } [ ] ( ) | \\`` must be escaped to be literal.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from learnai.kit.counters import Counters

#: The state of a walk that left the language. ``dfa.step(DEAD, b)`` is ``DEAD``.
DEAD = -1

_MAX_REPEAT = 100


class CharDFA:
    """A DFA over bytes with counted steps. See the module docstring.

    Args:
        transitions: ``{state: {byte: next_state}}``. States are non-negative ints. A
            byte is an int in ``[0, 256)``, or a one-byte ``bytes`` or one-character
            ASCII ``str`` for readability. A missing entry means ``DEAD``.
        start: the start state.
        accepting: the accepting states.

    Attributes:
        start, accepting (a frozenset), states (a sorted tuple), n_states.
        counters: ``steps``.
    """

    DEAD = DEAD

    def __init__(
        self,
        transitions: Mapping[int, Mapping[Any, int]],
        start: int,
        accepting: Iterable[int],
    ) -> None:
        table: dict[int, dict[int, int]] = {}
        for state, row in transitions.items():
            table[_state(state)] = {_byte(byte): _state(target) for byte, target in row.items()}
        self.start = _state(start)
        self.accepting = frozenset(_state(state) for state in accepting)
        mentioned = {self.start, *self.accepting, *table}
        mentioned.update(target for row in table.values() for target in row.values())
        self.states: tuple[int, ...] = tuple(sorted(mentioned))
        self._table = {state: table.get(state, {}) for state in self.states}
        self._live = self._states_reaching_acceptance()
        self.counters = Counters("steps")

    @property
    def n_states(self) -> int:
        return len(self.states)

    def with_framework(self, fw: Any) -> CharDFA:
        """The same automaton with fresh counters (it has no tensors to convert)."""
        twin = object.__new__(CharDFA)
        twin.__dict__.update(self.__dict__)
        twin.counters = Counters("steps")
        return twin

    # -- what a solution uses --------------------------------------------------------

    def step(self, state: int, byte: int) -> int:
        """The state after reading one byte from ``state``, or ``DEAD``. Counted.

        Raises ``ValueError`` for an unknown state or a byte outside ``[0, 256)``.
        """
        self.counters.add("steps")
        return self._next(state, byte)

    def walk(self, state: int, data: bytes) -> int:
        """Step through ``data`` from ``state``; stops as soon as the walk is ``DEAD``.

        Counts one step per byte consumed (the byte that kills the walk included, the
        bytes after it not). Walking no bytes returns ``state`` and counts nothing.
        """
        for byte in data:
            if state == DEAD:
                break
            state = self.step(state, byte)
        return state

    def is_accepting(self, state: int) -> bool:
        """Whether the bytes read so far form a complete string of the language."""
        return state in self.accepting

    # -- oracles for tests (not counted) ---------------------------------------------

    def is_live(self, state: int) -> bool:
        """Whether some continuation from ``state`` reaches an accepting state."""
        return state in self._live

    def accepts(self, data: bytes | str) -> bool:
        """Whether ``data`` is in the language. Not counted."""
        return self._run(data) in self.accepting

    def accepts_prefix(self, data: bytes | str) -> bool:
        """Whether ``data`` can be extended (possibly by nothing) to a string of the language."""
        return self._run(data) in self._live

    def trim(self) -> CharDFA:
        """An equivalent DFA without trap states: transitions into states that cannot
        reach acceptance are removed, so such walks become ``DEAD``. State numbers are
        kept. Raises ``ValueError`` if the language is empty."""
        if self.start not in self._live:
            raise ValueError("this DFA accepts nothing")
        table = {
            state: {byte: target for byte, target in row.items() if target in self._live}
            for state, row in self._table.items()
            if state in self._live
        }
        return CharDFA(table, self.start, self.accepting)

    # -- internals -------------------------------------------------------------------

    def _next(self, state: int, byte: int) -> int:
        if state == DEAD:
            return DEAD
        if state not in self._table:
            raise ValueError(f"unknown DFA state {state!r}")
        return self._table[state].get(_byte(byte), DEAD)

    def _run(self, data: bytes | str) -> int:
        state = self.start
        for byte in data.encode("utf-8") if isinstance(data, str) else data:
            state = self._next(state, byte)
        return state

    def _states_reaching_acceptance(self) -> frozenset[int]:
        sources: dict[int, set[int]] = {state: set() for state in self.states}
        for state, row in self._table.items():
            for target in row.values():
                sources[target].add(state)
        live, queue = set(self.accepting), deque(self.accepting)
        while queue:
            for source in sources[queue.popleft()]:
                if source not in live:
                    live.add(source)
                    queue.append(source)
        return frozenset(live)

    def __repr__(self) -> str:
        return f"CharDFA(states={self.n_states}, accepting={len(self.accepting)})"

    # -- from a regular expression ---------------------------------------------------

    @classmethod
    def from_regex(cls, pattern: str) -> CharDFA:
        """The minimal trimmed DFA of ``pattern`` (subset: see the module docstring).

        States are numbered in breadth-first order from the start state (0), following
        bytes in increasing order, so equal patterns give identical automata. Raises
        ``ValueError`` for syntax outside the subset.
        """
        nfa = _NFA()
        first, last = nfa.build(_Parser(pattern).parse())
        return cls(*_minimize(*_determinize(nfa, first, last)))


def _state(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"a DFA state must be a non-negative int, got {value!r}")
    return value


def _byte(value: Any) -> int:
    if isinstance(value, str) and len(value) == 1 and value.isascii():
        return ord(value)
    if isinstance(value, bytes) and len(value) == 1:
        return value[0]
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 256:
        raise ValueError(f"a byte must be an int in [0, 256), got {value!r}")
    return value


# ---------------------------------------------------------------------------------------
# Regex to DFA: parse to a tree, build an NFA (Thompson), make it deterministic (subset
# construction), drop states that cannot accept, merge equivalent states, renumber.
# ---------------------------------------------------------------------------------------

# Tree nodes: ("set", frozenset of bytes) reads one byte from the set; ("cat", [nodes]);
# ("alt", [nodes]); ("repeat", node, low, high) with high None for "no upper bound".
_Node = tuple

_ASCII = frozenset(range(0x80))
_DIGITS = frozenset(b"0123456789")
_WORD = frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACE = frozenset(b" \t\n\r\f\v")
_CLASSES = {"d": _DIGITS, "w": _WORD, "s": _SPACE}
_CONTROLS = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "0": "\0"}
_META = set(".^$*+?{}[]()|\\")


def _byte_set(values: Iterable[int]) -> _Node:
    return ("set", frozenset(values))


def _literal(char: str) -> _Node:
    """One character: a single byte for ASCII, the UTF-8 byte sequence otherwise."""
    return ("cat", [_byte_set([byte]) for byte in char.encode("utf-8")])


def _non_ascii() -> _Node:
    """Any one multi-byte UTF-8 character: a lead byte and its continuation bytes."""
    tail = _byte_set(range(0x80, 0xC0))
    return (
        "alt",
        [
            ("cat", [_byte_set(range(0xC2, 0xE0)), tail]),
            ("cat", [_byte_set(range(0xE0, 0xF0)), tail, tail]),
            ("cat", [_byte_set(range(0xF0, 0xF5)), tail, tail, tail]),
        ],
    )


def _any_char_except(excluded: Iterable[int]) -> _Node:
    """Any one character whose (ASCII) byte is not in ``excluded``."""
    return ("alt", [_byte_set(_ASCII - frozenset(excluded)), _non_ascii()])


class _Parser:
    """Recursive descent over the pattern: alternation > concatenation > repetition > atom."""

    def __init__(self, pattern: str) -> None:
        if not isinstance(pattern, str):
            raise TypeError("the pattern must be a str")
        self.pattern = pattern
        self.pos = 0

    def fail(self, why: str) -> ValueError:
        return ValueError(f"unsupported regex {self.pattern!r}: {why} (at position {self.pos})")

    def peek(self) -> str:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else ""

    def take(self) -> str:
        char = self.peek()
        if not char:
            raise self.fail("the pattern ends too early")
        self.pos += 1
        return char

    def parse(self) -> _Node:
        node = self.alternation()
        if self.pos < len(self.pattern):
            raise self.fail(f"unexpected {self.peek()!r}")
        return node

    def alternation(self) -> _Node:
        branches = [self.concatenation()]
        while self.peek() == "|":
            self.take()
            branches.append(self.concatenation())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def concatenation(self) -> _Node:
        items = []
        while self.peek() and self.peek() not in "|)":
            items.append(self.repetition())
        return ("cat", items)

    def repetition(self) -> _Node:
        node = self.atom()
        if self.peek() and self.peek() in "*+?{":
            char = self.take()
            if char == "{":
                low, high = self.bounds()
            else:
                low, high = {"*": (0, None), "+": (1, None), "?": (0, 1)}[char]
            if self.peek() and self.peek() in "*+?{":
                raise self.fail("lazy, possessive, and stacked quantifiers are not supported")
            node = ("repeat", node, low, high)
        return node

    def bounds(self) -> tuple[int, int | None]:
        end = self.pattern.find("}", self.pos)
        if end < 0:
            raise self.fail("unterminated {m,n}")
        low_text, comma, high_text = self.pattern[self.pos : end].partition(",")
        if not low_text.isdigit() or (high_text and not high_text.isdigit()):
            raise self.fail("a repetition must be {m}, {m,}, or {m,n}")
        low = int(low_text)
        high = low if not comma else (int(high_text) if high_text else None)
        if (high is not None and high < low) or max(low, high or 0) > _MAX_REPEAT:
            raise self.fail(f"repetition bounds must satisfy m <= n <= {_MAX_REPEAT}")
        self.pos = end + 1
        return low, high

    def atom(self) -> _Node:
        char = self.take()
        if char == "(":
            if self.peek() == "?":
                self.take()
                if self.take() != ":":
                    raise self.fail("only plain (...) and (?:...) groups are supported")
            node = self.alternation()
            if self.take() != ")":
                raise self.fail("expected ')'")
            return node
        if char == "[":
            return self.char_class()
        if char == ".":
            return _any_char_except(b"\n")
        if char == "\\":
            return self.escape()
        if char in "^$":
            raise self.fail("anchors are not supported; the whole string must match")
        if char in _META:
            raise self.fail(f"{char!r} must be escaped to be literal")
        return _literal(char)

    def escape(self) -> _Node:
        char = self.take()
        if char in _CLASSES:
            return _byte_set(_CLASSES[char])
        if char.lower() in _CLASSES:
            return _any_char_except(_CLASSES[char.lower()])
        return _byte_set([self.escaped_byte(char)])

    def escaped_byte(self, char: str) -> int:
        """The byte for an escape that stands for one byte (``char`` follows the backslash)."""
        if char in _CONTROLS:
            return ord(_CONTROLS[char])
        if char == "x":
            digits = self.pattern[self.pos : self.pos + 2]
            if len(digits) != 2 or any(d not in "0123456789abcdefABCDEF" for d in digits):
                raise self.fail("\\x must be followed by two hex digits")
            self.pos += 2
            return int(digits, 16)
        if char.isalnum() or not char.isascii():
            raise self.fail(f"unknown escape \\{char}")
        return ord(char)

    def char_class(self) -> _Node:
        negated = self.peek() == "^"
        if negated:
            self.take()
        members: set[int] = set()  # ASCII members, as bytes
        wide: list[str] = []  # non-ASCII members, as characters
        first = True
        while first or self.peek() != "]":
            first = False
            low = self.class_item()
            if isinstance(low, frozenset):  # \d, \w, \s
                members |= low
                continue
            if self.peek() == "-" and self.pattern[self.pos + 1 : self.pos + 2] not in ("]", ""):
                self.take()
                high = self.class_item()
                if isinstance(high, frozenset) or low > 0x7F or high > 0x7F or high < low:
                    raise self.fail("a range must run between two ASCII characters, low to high")
                members.update(range(low, high + 1))
            elif low > 0x7F:
                wide.append(chr(low))
            else:
                members.add(low)
        self.take()  # the closing bracket
        if negated:
            if wide:
                raise self.fail("a negated class may list ASCII characters only")
            return _any_char_except(members)
        branches = [_literal(char) for char in wide]
        if members:
            branches.insert(0, _byte_set(members))
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def class_item(self) -> int | frozenset[int]:
        """One class member: a code point, or a byte set for ``\\d``, ``\\w``, ``\\s``."""
        char = self.take()
        if char != "\\":
            return ord(char)
        char = self.take()
        if char in _CLASSES:
            return _CLASSES[char]
        if char.lower() in _CLASSES:
            raise self.fail(f"\\{char} is not supported inside a class")
        byte = self.escaped_byte(char)
        if byte > 0x7F:
            raise self.fail("raw bytes above 0x7F are not supported inside a class")
        return byte


class _NFA:
    """A nondeterministic automaton under construction: ``edges[s]`` is a list of
    ``(byte set, target)``, and ``free[s]`` lists the targets reachable without input."""

    def __init__(self) -> None:
        self.edges: list[list[tuple[frozenset[int], int]]] = []
        self.free: list[list[int]] = []

    def state(self) -> int:
        self.edges.append([])
        self.free.append([])
        return len(self.edges) - 1

    def build(self, node: _Node) -> tuple[int, int]:
        """Add ``node`` to the automaton; return its entry and exit states."""
        kind = node[0]
        first, last = self.state(), self.state()
        if kind == "set":
            self.edges[first].append((node[1], last))
        elif kind == "cat":
            here = first
            for item in node[1]:
                entry, exit_ = self.build(item)
                self.free[here].append(entry)
                here = exit_
            self.free[here].append(last)
        elif kind == "alt":
            for item in node[1]:
                entry, exit_ = self.build(item)
                self.free[first].append(entry)
                self.free[exit_].append(last)
        else:  # repeat: `low` required copies, then optional copies or a loop
            _, item, low, high = node
            here = first
            for _ in range(low):
                entry, here_next = self.build(item)
                self.free[here].append(entry)
                here = here_next
            if high is None:
                entry, exit_ = self.build(item)
                self.free[here].extend((entry, last))
                self.free[exit_].extend((entry, last))
            else:
                for _ in range(high - low):
                    entry, exit_ = self.build(item)
                    self.free[here].extend((entry, last))
                    here = exit_
                self.free[here].append(last)
        return first, last

    def closure(self, states: Iterable[int]) -> frozenset[int]:
        """``states`` plus everything reachable from them without reading input."""
        seen, stack = set(states), list(states)
        while stack:
            for target in self.free[stack.pop()]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)


_Table = dict[int, dict[int, int]]


def _determinize(nfa: _NFA, first: int, last: int) -> tuple[_Table, int, set[int]]:
    """Subset construction: each DFA state is the set of NFA states a prefix can reach."""
    start = nfa.closure([first])
    numbers = {start: 0}
    table: _Table = {}
    queue = deque([start])
    while queue:
        subset = queue.popleft()
        moves: dict[int, set[int]] = {}
        for state in subset:
            for byte_set, target in nfa.edges[state]:
                for byte in byte_set:
                    moves.setdefault(byte, set()).add(target)
        row = table.setdefault(numbers[subset], {})
        for byte, targets in moves.items():
            reached = nfa.closure(targets)
            if reached not in numbers:
                numbers[reached] = len(numbers)
                queue.append(reached)
            row[byte] = numbers[reached]
    accepting = {number for subset, number in numbers.items() if last in subset}
    return table, 0, accepting


def _minimize(table: _Table, start: int, accepting: set[int]) -> tuple[_Table, int, set[int]]:
    """Trim trap states, merge states no continuation can tell apart, and renumber."""
    trimmed = CharDFA(table, start, accepting).trim()
    table, accepting = trimmed._table, set(trimmed.accepting)

    # Moore's algorithm: start from {accepting, not accepting} and split a block
    # whenever two of its states go to different blocks on some byte.
    block = {state: int(state in accepting) for state in table}
    while True:
        signatures = {
            state: (block[state], tuple(sorted((b, block[t]) for b, t in row.items())))
            for state, row in table.items()
        }
        ids = {signature: i for i, signature in enumerate(sorted(set(signatures.values())))}
        refined = {state: ids[signature] for state, signature in signatures.items()}
        if len(ids) == len(set(block.values())):
            break
        block = refined

    # Renumber blocks breadth-first from the start, following bytes in increasing order.
    merged = {block[state]: {b: block[t] for b, t in row.items()} for state, row in table.items()}
    order = {block[start]: 0}
    queue = deque([block[start]])
    while queue:
        for _, target in sorted(merged[queue.popleft()].items()):
            if target not in order:
                order[target] = len(order)
                queue.append(target)
    final = {order[s]: {b: order[t] for b, t in row.items()} for s, row in merged.items()}
    return final, 0, {order[block[state]] for state in accepting}
