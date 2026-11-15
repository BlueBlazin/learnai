# BUG: alternation binds tighter than concatenation: the parser reads a sequence of alternatives, so ab|cd is compiled as a(b|c)d.
from collections import deque

MAX_REPEAT = 100
ASCII = frozenset(range(0x80))
SHORTHANDS = {
    "d": frozenset(b"0123456789"),
    "w": frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
    "s": frozenset(b" \t\n\r\f\v"),
}
CONTROLS = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B, "0": 0x00}
META = set(".^$*+?{}[]()|\\")

# Syntax tree: ("set", frozenset of bytes) reads one byte of the set; ("cat", [nodes]);
# ("alt", [nodes]); ("repeat", node, low, high), where high None means "no upper bound".
CONT = ("set", frozenset(range(0x80, 0xC0)))  # a UTF-8 continuation byte
MULTIBYTE = [  # one non-ASCII character: a lead byte and its continuation bytes
    ("cat", [("set", frozenset(range(0xC2, 0xE0))), CONT]),
    ("cat", [("set", frozenset(range(0xE0, 0xF0))), CONT, CONT]),
    ("cat", [("set", frozenset(range(0xF0, 0xF5))), CONT, CONT, CONT]),
]


def any_char_except(excluded):
    """One whole character whose ASCII byte, if it is ASCII, is not excluded."""
    return ("alt", [("set", ASCII - frozenset(excluded)), *MULTIBYTE])


class Parser:
    """Recursive descent: alternation > concatenation > repetition > atom."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.pos = 0

    def peek(self):
        return self.pattern[self.pos] if self.pos < len(self.pattern) else ""

    def take(self):
        if self.pos >= len(self.pattern):
            raise ValueError(f"{self.pattern!r} ends too early")
        self.pos += 1
        return self.pattern[self.pos - 1]

    def error(self, why):
        return ValueError(f"{self.pattern!r}: {why} at position {self.pos}")

    def parse(self):
        node = self.alternation()
        if self.pos < len(self.pattern):  # only an unmatched ")" stops the descent early
            raise self.error("unbalanced )")
        return node

    def alternation(self):
        pieces = []
        while self.peek() not in ("", ")"):
            pieces.append(self.choice())
        return ("cat", pieces)

    def choice(self):
        empty = ("cat", [])
        branches = [empty if self.peek() == "|" else self.repetition()]
        while self.peek() == "|":
            self.take()
            branches.append(empty if self.peek() in ("", "|", ")") else self.repetition())
        return ("alt", branches)

    def repetition(self):
        node = self.atom()
        if self.peek() in ("*", "+", "?", "{"):
            mark = self.take()
            low, high = (
                self.bounds()
                if mark == "{"
                else {"*": (0, None), "+": (1, None), "?": (0, 1)}[mark]
            )
            if self.peek() in ("*", "+", "?", "{"):
                raise self.error("a quantifier cannot follow a quantifier")
            node = ("repeat", node, low, high)
        return node

    def bounds(self):
        end = self.pattern.find("}", self.pos)
        if end < 0:
            raise self.error("unterminated {")
        low, comma, high = self.pattern[self.pos : end].partition(",")
        digits = "0123456789"
        if not low or any(c not in digits for c in low + high):
            raise self.error("a bound must be {m}, {m,}, or {m,n}")
        low = int(low)
        high = low if not comma else (int(high) if high else None)
        if (high is not None and high < low) or max(low, high or 0) > MAX_REPEAT:
            raise self.error(f"bounds must satisfy m <= n <= {MAX_REPEAT}")
        self.pos = end + 1
        return low, high

    def atom(self):
        char = self.take()
        if char == "(":
            if self.peek() == "?":
                self.take()
                if self.take() != ":":
                    raise self.error("only (...) and (?:...) groups are supported")
            node = self.alternation()
            if self.take() != ")":
                raise self.error("expected )")
            return node
        if char == "[":
            return self.char_class()
        if char == ".":
            return any_char_except(b"\n")
        if char == "\\":
            char = self.take()
            if char in SHORTHANDS:
                return ("set", SHORTHANDS[char])
            if char.lower() in SHORTHANDS:  # \D, \W, \S
                return any_char_except(SHORTHANDS[char.lower()])
            return ("set", frozenset([self.escaped_byte(char)]))
        if char in META:
            raise self.error(f"unexpected {char!r}")
        return ("set", frozenset([ord(char)]))

    def escaped_byte(self, char):
        """The byte an escape stands for; `char` is the character after the backslash."""
        if char in CONTROLS:
            return CONTROLS[char]
        if char == "x":
            digits = self.pattern[self.pos : self.pos + 2]
            if len(digits) != 2 or any(d not in "0123456789abcdefABCDEF" for d in digits):
                raise self.error("\\x needs two hex digits")
            self.pos += 2
            return int(digits, 16)
        if char.isalnum():
            raise self.error(f"unknown escape \\{char}")
        return ord(char)

    def char_class(self):
        negated = self.peek() == "^"
        if negated:
            self.take()
        members = set()
        while True:
            low = self.class_item()
            if isinstance(low, frozenset):  # \d, \w, \s
                members |= low
            elif self.peek() == "-" and self.pattern[self.pos + 1 : self.pos + 2] not in ("]", ""):
                self.take()
                high = self.class_item()
                if isinstance(high, frozenset) or high < low:
                    raise self.error("a range must run from a lower to a higher character")
                members.update(range(low, high + 1))
            else:
                members.add(low)
            if self.peek() == "]":
                break
        self.take()
        return any_char_except(members) if negated else ("set", frozenset(members))

    def class_item(self):
        """One class member: a byte, or a byte set for \\d, \\w, \\s."""
        char = self.take()
        if char != "\\":
            return ord(char)
        char = self.take()
        if char in SHORTHANDS:
            return SHORTHANDS[char]
        if char.lower() in SHORTHANDS:
            raise self.error(f"\\{char} is not supported inside a class")
        return self.escaped_byte(char)


class NFA:
    """Thompson's construction. edges[s]: list of (byte set, target); free[s]: epsilon targets."""

    def __init__(self):
        self.edges = []
        self.free = []

    def state(self):
        self.edges.append([])
        self.free.append([])
        return len(self.edges) - 1

    def build(self, node):
        """Add the fragment for `node`; return its (entry, exit) states."""
        entry, exit_ = self.state(), self.state()
        kind = node[0]
        if kind == "set":
            self.edges[entry].append((node[1], exit_))
        elif kind == "cat":
            here = entry
            for part in node[1]:
                first, last = self.build(part)
                self.free[here].append(first)
                here = last
            self.free[here].append(exit_)
        elif kind == "alt":
            for part in node[1]:
                first, last = self.build(part)
                self.free[entry].append(first)
                self.free[last].append(exit_)
        else:
            _, part, low, high = node
            here = entry
            for _ in range(low):  # the required copies, in a row
                first, last = self.build(part)
                self.free[here].append(first)
                here = last
            if high is None:  # then a loop ...
                first, last = self.build(part)
                self.free[here] += [first, exit_]
                self.free[last] += [first, exit_]
            else:  # ... or high - low optional copies, each of which may be skipped
                for _ in range(high - low):
                    first, last = self.build(part)
                    self.free[here] += [first, exit_]
                    here = last
                self.free[here].append(exit_)
        return entry, exit_

    def closure(self, states):
        """`states` and everything reachable from them without reading a byte."""
        seen, stack = set(states), list(states)
        while stack:
            for target in self.free[stack.pop()]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)


def determinize(nfa, entry, exit_):
    """Subset construction: a DFA state is the set of NFA states a prefix can reach."""
    start = nfa.closure([entry])
    number = {start: 0}
    table = {}
    queue = deque([start])
    while queue:
        subset = queue.popleft()
        moves = {}
        for state in subset:
            for byte_set, target in nfa.edges[state]:
                for byte in byte_set:
                    moves.setdefault(byte, set()).add(target)
        row = table[number[subset]] = {}
        seen = {}  # many bytes lead to the same target set: close each set once
        for byte, targets in moves.items():
            key = frozenset(targets)
            if key not in seen:
                reached = nfa.closure(key)
                if reached not in number:
                    number[reached] = len(number)
                    queue.append(reached)
                seen[key] = number[reached]
            row[byte] = seen[key]
    accepting = {n for subset, n in number.items() if exit_ in subset}
    return table, accepting


def minimize(table, accepting):
    """Moore's algorithm: refine {accepting, rest} until no block can be split."""
    block = {state: int(state in accepting) for state in table}
    count = len(set(block.values()))
    while True:
        # Two states stay together only if they are in one block now and every byte
        # leads them to the same block (a missing byte counts as "rejects").
        signature = {
            state: (block[state], tuple(sorted((byte, block[t]) for byte, t in row.items())))
            for state, row in table.items()
        }
        ids = {}
        block = {state: ids.setdefault(sig, len(ids)) for state, sig in signature.items()}
        if len(ids) == count:
            break
        count = len(ids)
    merged = {block[s]: {byte: block[t] for byte, t in row.items()} for s, row in table.items()}
    return merged, block[0], {block[s] for s in accepting}


def compile_regex(pattern: str) -> tuple[dict[int, dict[int, int]], int, set[int]]:
    nfa = NFA()
    entry, exit_ = nfa.build(Parser(pattern).parse())
    # Every NFA state lies on a path to the exit, so every subset state can still
    # accept: the DFA has no dead state to trim, and no unreachable one by construction.
    return minimize(*determinize(nfa, entry, exit_))
