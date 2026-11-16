"""Grammar: a context-free grammar in a subset of GBNF, parsed, with an oracle.

GBNF is the grammar notation of llama.cpp (``grammars/README.md`` in its repository):

    g = Grammar.parse('''
        root  ::= "[" ws (value ("," ws value)*)? "]"
        value ::= [0-9]+ | "true" | "false" | root      # nested lists
        ws    ::= " "?
    ''')
    g.accepts("[1, [true]]")             # True: the whole text is in the language
    g.accepts_prefix("[1, [tr")          # True: it can still become a member
    g.accepts_prefix("[1,,")             # False: nothing that starts like this is

A regular expression cannot count brackets; a grammar can, which is why structured
output for nested formats is specified with one. ``Grammar`` does the parsing of the
notation and answers the two membership questions, so that a problem can be about the
part that matters (deciding, token by token, what may come next) and still be judged
exactly.

The supported subset
--------------------
From the llama.cpp documentation, as written there:

* a rule is ``name ::= ...``; the language is what the rule ``root`` matches. Names
  are dashed lowercase words (``check-mate``); digits are accepted after the first
  letter;
* terminals are Unicode code points: string literals (``"O-O"``) and character sets
  (``[1-9]``, ``[NBKQR]``, ``[^\\n]`` negated, ranges of any code points);
* escapes ``\\n \\t \\r \\\\ \\" \\[ \\]`` and ``\\xXX``, ``\\uXXXX``, ``\\UXXXXXXXX``;
* sequences, alternatives ``|``, grouping ``( ... )``;
* repetition of the preceding symbol or group: ``*``, ``+``, ``?``, ``{m}``, ``{m,}``,
  ``{m,n}``. After a string literal it repeats the whole literal;
* ``#`` starts a comment;
* a rule ends at the end of its line, except inside parentheses and directly after
  ``|``, where it continues on the next line.

Not supported, each a ``ValueError`` that says so: token references (``<[1000]>``,
``<think>``, ``!<...>``), and ``.`` (the documentation uses it in an example without
defining it; write ``[^\\n]`` or the set you mean). A rule that is used but never
defined, a rule defined twice, a missing ``root``, and a rule that can never finish
(``a ::= "x" a``) are errors too.

Recursion of every kind is allowed, including **left recursion** (``expr ::= expr "+"
term | term``) and rules that match the empty string. The oracle handles them;
``g.left_recursive`` names the rules that can begin with themselves, because a matcher
written as a naive stack machine loops on exactly those.

The parsed form
---------------
``g.rules`` maps every rule name to its alternatives; an alternative is a tuple of
symbols; a symbol is ``Chars`` (one character out of a set) or ``Ref`` (another rule):

    g.rules["value"]    ((Ref("value_1"),), (Chars 't', Chars 'r', Chars 'u', Chars 'e'), ...)

Everything else is rewritten into that shape, the way llama.cpp does it internally. A
literal becomes one ``Chars`` per character. A group becomes a new rule. Repetition
becomes new rules: ``x*`` is ``r ::= x r | (empty)``, ``x+`` is ``x`` followed by
that, ``x?`` is ``r ::= x | (empty)``, and ``x{m,n}`` is ``m`` copies of ``x``
followed by ``n - m`` nested optional ones. Generated rules are named after the rule
they come from, with an underscore and a number (``value_1``): a name the notation
itself cannot produce. ``g.rules`` is in definition order, ``root`` included wherever
it was written. An alternative may be empty (``()``).

The oracle
----------
``accepts(text)`` and ``accepts_prefix(text)`` take a ``str`` or ``bytes`` (UTF-8).
A byte string that ends in the middle of a character is a prefix when some character
that starts with those bytes would be accepted there; bytes that are not UTF-8 are
never accepted. Both run an Earley recognizer over ``g.rules``, which is exact for
every grammar this class accepts. They are for tests: a solution that calls them has
not written a matcher, so tell learners not to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The largest Unicode code point.
MAX_CODE_POINT = 0x10FFFF
#: Bounds of a repetition count, to keep generated grammars small.
MAX_REPEAT = 100


@dataclass(frozen=True)
class Chars:
    """One character out of a set of code points.

    ``ranges`` are inclusive ``(low, high)`` code point pairs, sorted and not
    overlapping; with ``negated`` the symbol matches every character *not* in them.
    """

    ranges: tuple[tuple[int, int], ...]
    negated: bool = False

    def matches(self, char: str | int) -> bool:
        """Whether the character (a one-character string or a code point) is in the set."""
        code = ord(char) if isinstance(char, str) else int(char)
        inside = any(low <= code <= high for low, high in self.ranges)
        return inside != self.negated

    def overlaps(self, low: int, high: int) -> bool:
        """Whether any code point in ``[low, high]`` is in the set."""
        if not self.negated:
            return any(a <= high and low <= b for a, b in self.ranges)
        at = low  # the smallest code point not yet known to be excluded
        for a, b in self.ranges:
            if at < a:
                break
            at = max(at, b + 1)
        return at <= high

    def __repr__(self) -> str:
        def show(code: int) -> str:
            char = chr(code)
            return char if char.isprintable() and char not in "[]^-\\" else f"\\u{code:04X}"

        inner = "".join(show(a) if a == b else f"{show(a)}-{show(b)}" for a, b in self.ranges)
        return f"Chars([{'^' if self.negated else ''}{inner}])"


@dataclass(frozen=True)
class Ref:
    """A reference to the rule ``name``."""

    name: str


Symbol = Chars | Ref
Alternative = tuple[Symbol, ...]


def _merge(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for low, high in sorted(ranges):
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return tuple(merged)


class Grammar:
    """A parsed grammar. Build one with ``Grammar.parse(text)``; see the module docstring.

    Attributes:
        rules: ``{name: (alternative, ...)}``, each alternative a tuple of ``Chars``
            and ``Ref``; generated helper rules included.
        root: ``"root"``.
        source: the text it was parsed from.
        nullable: names of the rules that can match the empty string.
        left_recursive: names of the rules that can begin with themselves.
    """

    root = "root"

    def __init__(self, rules: dict[str, tuple[Alternative, ...]], source: str = "") -> None:
        self.rules = rules
        self.source = source
        _validate(rules)
        self.nullable = _nullable(rules)
        self.left_recursive = _left_recursive(rules, self.nullable)

    @classmethod
    def parse(cls, text: str) -> Grammar:
        """Parse GBNF ``text``. Raises ``ValueError`` with the line for anything outside
        the supported subset."""
        if not isinstance(text, str):
            raise TypeError("Grammar.parse takes the grammar as a str")
        return cls(_Parser(text).parse(), text)

    # -- the oracle --------------------------------------------------------------------------

    def accepts(self, text: str | bytes) -> bool:
        """Whether ``text``, all of it, is in the language."""
        chars, rest = _decode(text)
        if chars is None or rest is not None:
            return False
        items = self._run(chars)
        return items is not None and any(
            name == self.root and origin == 0 and dot == len(self.rules[name][alt])
            for name, alt, dot, origin in items
        )

    def accepts_prefix(self, text: str | bytes) -> bool:
        """Whether ``text`` can be extended (by nothing, possibly) to a member of the language."""
        chars, rest = _decode(text)
        if chars is None:
            return False
        items = self._run(chars)
        if items is None:
            return False
        if rest is None:
            return True
        # The text ends inside a character: is any character with these first bytes next?
        low, high = rest
        for name, alt, dot, _ in items:
            body = self.rules[name][alt]
            if dot < len(body) and isinstance(body[dot], Chars) and body[dot].overlaps(low, high):
                return True
        return False

    def _run(self, chars: str) -> set[tuple[str, int, int, int]] | None:
        """The Earley item set after ``chars``, or None if no parse survives them.

        An item ``(rule, alternative, dot, origin)`` says: this alternative has matched
        up to ``dot``, starting at position ``origin``.
        """
        rules, nullable = self.rules, self.nullable
        chart: list[set[tuple[str, int, int, int]]] = []
        current = {(self.root, alt, 0, 0) for alt in range(len(rules[self.root]))}
        for position in range(len(chars) + 1):
            # Close the set: predict the rules that may start here, and complete the
            # ones that have ended.
            todo = list(current)
            while todo:
                name, alt, dot, origin = todo.pop()
                body = rules[name][alt]
                new: list[tuple[str, int, int, int]] = []
                if dot == len(body):
                    waiting = chart[origin] if origin < position else current
                    for other in list(waiting):
                        o_name, o_alt, o_dot, o_origin = other
                        o_body = rules[o_name][o_alt]
                        if o_dot < len(o_body) and o_body[o_dot] == Ref(name):
                            new.append((o_name, o_alt, o_dot + 1, o_origin))
                elif isinstance(body[dot], Ref):
                    target = body[dot].name
                    new.extend((target, a, 0, position) for a in range(len(rules[target])))
                    if target in nullable:  # it may match nothing: step over it as well
                        new.append((name, alt, dot + 1, origin))
                for item in new:
                    if item not in current:
                        current.add(item)
                        todo.append(item)
            chart.append(current)
            if position == len(chars):
                return current
            char = chars[position]
            scanned = set()
            for name, alt, dot, origin in current:
                body = rules[name][alt]
                if dot < len(body) and isinstance(body[dot], Chars) and body[dot].matches(char):
                    scanned.add((name, alt, dot + 1, origin))
            if not scanned:
                return None
            current = scanned
        return None  # not reached

    def __repr__(self) -> str:
        return f"Grammar({len(self.rules)} rules, root={self.root!r})"


# -- text and bytes ------------------------------------------------------------------------


def _decode(text: str | bytes) -> tuple[str | None, tuple[int, int] | None]:
    """``(characters, pending)``: the complete characters of ``text`` and, when it ends
    inside one, the range of code points that begin with the leftover bytes.
    ``(None, None)`` for bytes that are not UTF-8."""
    if isinstance(text, str):
        return text, None
    data = bytes(text)
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError as error:
        head, tail = data[: error.start], data[error.start :]
        pending = _partial_range(tail)
        if pending is None or error.end != len(data):
            return None, None
        try:
            return head.decode("utf-8"), pending
        except UnicodeDecodeError:
            return None, None


def _partial_range(tail: bytes) -> tuple[int, int] | None:
    """The code points whose UTF-8 encoding starts with ``tail`` (1 to 3 bytes), as an
    inclusive range, or None when no character starts that way."""
    if not 1 <= len(tail) <= 3:
        return None
    lead = tail[0]
    if 0xC2 <= lead <= 0xDF:
        length, bits = 2, lead & 0x1F
    elif 0xE0 <= lead <= 0xEF:
        length, bits = 3, lead & 0x0F
    elif 0xF0 <= lead <= 0xF4:
        length, bits = 4, lead & 0x07
    else:
        return None
    if len(tail) >= length or any(not 0x80 <= b <= 0xBF for b in tail[1:]):
        return None
    low = high = bits
    for byte in tail[1:]:
        low = high = (low << 6) | (byte & 0x3F)
    missing = length - len(tail)
    low, high = low << (6 * missing), (high << (6 * missing)) | ((1 << (6 * missing)) - 1)
    # Keep to characters that exist: no overlong forms, no surrogates, nothing above U+10FFFF.
    floor = {2: 0x80, 3: 0x800, 4: 0x10000}[length]
    low, high = max(low, floor), min(high, MAX_CODE_POINT)
    if low <= 0xDFFF and high >= 0xD800:  # the range touches the surrogates
        if low >= 0xD800 and high <= 0xDFFF:
            return None
        low, high = (low, 0xD7FF) if low < 0xD800 else (0xE000, high)
    return (low, high) if low <= high else None


# -- properties of a grammar ---------------------------------------------------------------


def _validate(rules: dict[str, tuple[Alternative, ...]]) -> None:
    if Grammar.root not in rules:
        raise ValueError("the grammar has no `root` rule")
    for name, alternatives in rules.items():
        for alternative in alternatives:
            for symbol in alternative:
                if isinstance(symbol, Ref) and symbol.name not in rules:
                    raise ValueError(f"rule `{name}` uses `{symbol.name}`, which is not defined")
    # A rule is productive when some alternative consists of characters and productive
    # rules only: it can finish. A rule that cannot would make "can this prefix still
    # be completed?" false for text that looks fine, so it is refused here.
    productive: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, alternatives in rules.items():
            if name not in productive and any(
                all(isinstance(s, Chars) or s.name in productive for s in alternative)
                for alternative in alternatives
            ):
                productive.add(name)
                changed = True
    stuck = [name for name in rules if name not in productive]
    if stuck:
        raise ValueError(
            f"rule `{stuck[0]}` can never finish: every alternative leads back into a rule "
            "that cannot end"
        )


def _nullable(rules: dict[str, tuple[Alternative, ...]]) -> frozenset[str]:
    nullable: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, alternatives in rules.items():
            if name not in nullable and any(
                all(isinstance(s, Ref) and s.name in nullable for s in alternative)
                for alternative in alternatives
            ):
                nullable.add(name)
                changed = True
    return frozenset(nullable)


def _left_recursive(
    rules: dict[str, tuple[Alternative, ...]], nullable: frozenset[str]
) -> frozenset[str]:
    """Rules that can derive a string of symbols beginning with themselves."""
    first: dict[str, set[str]] = {name: set() for name in rules}
    for name, alternatives in rules.items():
        for alternative in alternatives:
            for symbol in alternative:
                if isinstance(symbol, Chars):
                    break
                first[name].add(symbol.name)
                if symbol.name not in nullable:
                    break
    found = set()
    for name in rules:
        seen, todo = set(), list(first[name])
        while todo:
            other = todo.pop()
            if other not in seen:
                seen.add(other)
                todo.extend(first[other])
        if name in seen:
            found.add(name)
    return frozenset(found)


# -- the parser ----------------------------------------------------------------------------

_NAME_START = "abcdefghijklmnopqrstuvwxyz"
_NAME_REST = _NAME_START + "0123456789-"
_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "[": "[", "]": "]"}
_HEX_ESCAPES = {"x": 2, "u": 4, "U": 8}


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.rules: dict[str, tuple[Alternative, ...]] = {}
        self.generated: dict[str, int] = {}

    def fail(self, why: str) -> ValueError:
        line = self.text.count("\n", 0, self.pos) + 1
        return ValueError(f"grammar, line {line}: {why}")

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def space(self, newlines: bool) -> None:
        """Skip blanks and comments, and line ends too when ``newlines``."""
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char == "#":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            elif char in " \t" or (newlines and char in "\r\n"):
                self.pos += 1
            else:
                break

    def parse(self) -> dict[str, tuple[Alternative, ...]]:
        self.space(newlines=True)
        while self.pos < len(self.text):
            self.rule()
            self.space(newlines=True)
        if not self.rules:
            raise self.fail("the grammar is empty")
        return self.rules

    def name(self) -> str:
        start = self.pos
        if self.peek() not in _NAME_START or not self.peek():
            raise self.fail(
                f"expected a rule name (a dashed lowercase word), found {self.peek()!r}"
            )
        while self.peek() and self.peek() in _NAME_REST:
            self.pos += 1
        return self.text[start : self.pos]

    def rule(self) -> None:
        name = self.name()
        self.space(newlines=False)
        if self.text[self.pos : self.pos + 3] != "::=":
            raise self.fail(f"expected `::=` after the rule name `{name}`")
        self.pos += 3
        self.space(newlines=True)
        if name in self.rules:
            raise self.fail(f"rule `{name}` is defined twice")
        self.rules[name] = ()  # reserves the rule's place in definition order
        self.rules[name] = self.alternatives(name, nested=False)
        if self.peek() and self.peek() not in "\r\n":
            raise self.fail(f"unexpected {self.peek()!r} in rule `{name}`")

    def alternatives(self, rule: str, nested: bool) -> tuple[Alternative, ...]:
        found = [self.sequence(rule, nested)]
        while self.peek() == "|":
            self.pos += 1
            self.space(newlines=True)  # a line break directly after `|` continues the rule
            found.append(self.sequence(rule, nested))
        return tuple(found)

    def sequence(self, rule: str, nested: bool) -> Alternative:
        out: list[Symbol] = []
        last: list[Symbol] | None = None  # the symbols a repetition operator would repeat
        while True:
            char = self.peek()
            if char == '"':
                self.pos += 1
                last = []
                while self.peek() != '"':
                    if not self.peek() or self.peek() in "\r\n":
                        raise self.fail("a string literal is not closed")
                    code = self.char()
                    last.append(Chars(((code, code),)))
                self.pos += 1
            elif char == "[":
                last = [self.char_set()]
            elif char == "(":
                self.pos += 1
                self.space(newlines=True)
                inner = self.alternatives(rule, nested=True)
                if self.peek() != ")":
                    raise self.fail("a group is not closed: expected `)`")
                self.pos += 1
                last = [Ref(self.new_rule(rule, inner))]
            elif char and char in _NAME_START:
                last = [Ref(self.name())]
            elif char and char in "*+?{":
                if last is None:
                    raise self.fail(f"`{char}` has nothing to repeat")
                low, high = self.bounds()
                del out[len(out) - len(last) :]
                out.extend(self.repeat(rule, last, low, high))
                last = None
                self.space(newlines=nested)
                continue
            elif char == ".":
                raise self.fail(
                    "`.` is not supported: the GBNF documentation does not define it; "
                    "write the character set you mean, for example [^\\n]"
                )
            elif char and char in "<!":
                raise self.fail("token references (`<...>`, `!<...>`) are not supported")
            else:
                break
            out.extend(last)
            self.space(newlines=nested)
        return tuple(out)

    def char(self) -> int:
        """One character of a literal or a set, as a code point; escapes resolved."""
        char = self.peek()
        self.pos += 1
        if char != "\\":
            return ord(char)
        kind = self.peek()
        self.pos += 1
        if kind in _SIMPLE_ESCAPES:
            return ord(_SIMPLE_ESCAPES[kind])
        if kind in _HEX_ESCAPES:
            digits = self.text[self.pos : self.pos + _HEX_ESCAPES[kind]]
            if len(digits) != _HEX_ESCAPES[kind] or any(
                d not in "0123456789abcdefABCDEF" for d in digits
            ):
                raise self.fail(f"`\\{kind}` needs {_HEX_ESCAPES[kind]} hexadecimal digits")
            self.pos += len(digits)
            code = int(digits, 16)
            if code > MAX_CODE_POINT:
                raise self.fail(f"\\{kind}{digits} is not a Unicode code point")
            return code
        raise self.fail(f"unknown escape `\\{kind}`")

    def char_set(self) -> Chars:
        self.pos += 1  # [
        negated = self.peek() == "^"
        if negated:
            self.pos += 1
        ranges: list[tuple[int, int]] = []
        while self.peek() != "]":
            if not self.peek() or self.peek() in "\r\n":
                raise self.fail("a character set is not closed: expected `]`")
            low = high = self.char()
            if self.peek() == "-" and self.text[self.pos + 1 : self.pos + 2] not in ("]", ""):
                self.pos += 1
                high = self.char()
                if high < low:
                    raise self.fail(f"the range {chr(low)!r}-{chr(high)!r} is backwards")
            ranges.append((low, high))
        self.pos += 1
        if not ranges and not negated:
            raise self.fail("an empty character set `[]` matches nothing")
        return Chars(_merge(ranges), negated)

    def bounds(self) -> tuple[int, int | None]:
        """The repetition operator at the cursor as ``(min, max)``; max None: unbounded."""
        char = self.peek()
        self.pos += 1
        if char in "*+?":
            return {"*": (0, None), "+": (1, None), "?": (0, 1)}[char]
        end = self.text.find("}", self.pos)
        body = self.text[self.pos : end] if end != -1 else ""
        parts = [part.strip() for part in body.split(",")]
        valid = end != -1 and len(parts) in (1, 2) and parts[0].isdigit()
        if valid and len(parts) == 2 and parts[1] and not parts[1].isdigit():
            valid = False
        if not valid:
            raise self.fail("a repetition is written {m}, {m,}, or {m,n}")
        self.pos = end + 1
        low = int(parts[0])
        high = low if len(parts) == 1 else (int(parts[1]) if parts[1] else None)
        if high is not None and high < low:
            raise self.fail(f"the repetition {{{low},{high}}} has its bounds backwards")
        if max(low, high or 0) > MAX_REPEAT:
            raise self.fail(f"repetition counts above {MAX_REPEAT} are not supported")
        return low, high

    def repeat(self, rule: str, item: list[Symbol], low: int, high: int | None) -> list[Symbol]:
        """``item`` repeated ``low`` to ``high`` times, as symbols (see the module docstring)."""
        out = list(item) * low
        body = tuple(item)
        if high is None:
            # r ::= item r | (empty)
            name = self.new_rule(rule, ())
            self.rules[name] = ((*body, Ref(name)), ())
            out.append(Ref(name))
        elif high > low:
            # r1 ::= item r2 | (empty);  r2 ::= item r3 | (empty);  ...  rk ::= item | (empty)
            names = [self.new_rule(rule, ()) for _ in range(high - low)]
            for index, name in enumerate(names):
                tail = (Ref(names[index + 1]),) if index + 1 < len(names) else ()
                self.rules[name] = ((*body, *tail), ())
            out.append(Ref(names[0]))
        return out

    def new_rule(self, rule: str, alternatives: Any) -> str:
        self.generated[rule] = self.generated.get(rule, 0) + 1
        name = f"{rule}_{self.generated[rule]}"
        self.rules[name] = tuple(alternatives)
        return name
