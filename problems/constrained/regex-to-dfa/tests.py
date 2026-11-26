"""Tests for regex-to-dfa.

Three independent sources of truth:

* the kit's `CharDFA.from_regex`, which is the minimal automaton of the same subset.
  Two DFAs are compared exactly, not by sampling: a breadth-first walk over pairs of
  states finds the shortest byte string that one accepts and the other rejects, if
  there is one (`shortest_difference`);
* Python's `re.fullmatch(pattern, text, re.ASCII)` on generated texts: strings read
  off the automaton, corrupted copies of them, and random strings over the pattern's
  alphabet (`check_against_re`);
* hand-written tables and state counts for the statement's examples.

Every pattern also goes through the structural checks of the statement: the format of
the result, no unreachable state, no state that cannot accept, and the minimal number
of states.
"""

import functools
import re
from collections import deque

from learnai.judge import case, check
from learnai.kit import CharDFA

FN = "compile_regex"


# -- reading and checking a result -------------------------------------------------------


def checked(result, pattern):
    """The result of compile_regex as (transitions, start, accepting), format verified."""
    what = FN
    check.that(
        isinstance(result, tuple) and len(result) == 3,
        f"{what} must return a tuple (transitions, start, accepting)",
    )
    table, start, accepting = result
    check.that(isinstance(table, dict), f"{what}: transitions must be a dict")
    check.that(
        isinstance(accepting, (set, frozenset)),
        f"{what}: accepting must be a set",
        f"got {type(accepting).__name__}",
    )

    def is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    check.that(
        all(is_int(s) for s in table) and all(isinstance(row, dict) for row in table.values()),
        f"{what}: transitions must map int states to dicts",
    )
    check.that(
        is_int(start) and start in table,
        f"{what}: the start state has no entry in transitions",
        f"start = {start!r}; every state needs an entry, possibly an empty dict",
    )
    check.that(
        all(s in table for s in accepting),
        f"{what}: an accepting state has no entry in transitions",
        f"accepting = {sorted(accepting)!r}, states = {sorted(table)!r}",
    )
    for state, row in table.items():
        for byte, target in row.items():
            check.that(
                is_int(byte) and 0 <= byte < 256,
                f"{what}: a transition is not labeled with an int in [0, 256)",
                f"state {state}: label {byte!r}",
            )
            check.that(
                target in table,
                f"{what}: a transition leads to a state without an entry in transitions",
                f"state {state}, byte {byte}: target {target!r}",
            )
    return table, start, set(accepting)


def shortest_difference(a, b):
    """The shortest bytes accepted by exactly one of two DFAs, or None if they are equal.

    Each DFA is (transitions, start, accepting); None stands for "rejected for good".
    """
    (ta, sa, fa), (tb, sb, fb) = a, b
    seen = {(sa, sb): b""}
    queue = deque([(sa, sb)])
    while queue:
        pa, pb = queue.popleft()
        path = seen[(pa, pb)]
        if (pa in fa) != (pb in fb):
            return path
        row_a = ta[pa] if pa is not None else {}
        row_b = tb[pb] if pb is not None else {}
        for byte in sorted(set(row_a) | set(row_b)):
            pair = (row_a.get(byte), row_b.get(byte))
            if pair not in seen:
                seen[pair] = path + bytes([byte])
                queue.append(pair)
    return None


def accepts(dfa, data):
    table, state, accepting = dfa
    for byte in data:
        state = table[state].get(byte)
        if state is None:
            return False
    return state in accepting


@functools.cache
def kit_dfa(pattern):
    dfa = CharDFA.from_regex(pattern)
    return dfa._table, dfa.start, set(dfa.accepting)


def show(data):
    """Bytes for a failure detail: as text when they are valid UTF-8."""
    try:
        return f"{data.decode('utf-8')!r} (bytes {data!r})"
    except UnicodeDecodeError:
        return f"bytes {data!r}"


def check_language(dfa, pattern):
    """The automaton accepts exactly the language of the pattern."""
    witness = shortest_difference(dfa, kit_dfa(pattern))
    if witness is None:
        return
    detail = f"pattern: {pattern!r}\nshortest such string: {show(witness)}"
    if accepts(dfa, witness):
        check.that(
            False, f"{FN}: the automaton accepts a string the pattern does not match", detail
        )
    check.that(False, f"{FN}: the automaton rejects a string the pattern matches", detail)


def check_structure(dfa, pattern):
    """No unreachable state, no state that cannot accept, and no state more than needed."""
    table, start, accepting = dfa
    reachable, stack = {start}, [start]
    while stack:
        for target in table[stack.pop()].values():
            if target not in reachable:
                reachable.add(target)
                stack.append(target)
    check.that(
        len(reachable) == len(table),
        f"{FN}: {len(table) - len(reachable)} of the {len(table)} states cannot be reached "
        "from the start state",
        f"pattern: {pattern!r}\nunreachable: {sorted(set(table) - reachable)}",
    )
    live = set(accepting)
    changed = True
    while changed:
        changed = False
        for state, row in table.items():
            if state not in live and any(t in live for t in row.values()):
                live.add(state)
                changed = True
    check.that(
        len(live) == len(table),
        f"{FN}: {len(table) - len(live)} of the {len(table)} states cannot reach an "
        "accepting state",
        f"pattern: {pattern!r}\ndead states: {sorted(set(table) - live)}; leave the "
        "transition out instead",
    )
    minimal = len(kit_dfa(pattern)[0])
    check.that(
        len(table) == minimal,
        f"{FN}: the automaton has {len(table)} states, expected {minimal} (the minimal number)",
        f"pattern: {pattern!r}\nthe minimal automaton of this language has {minimal} states",
    )


def compile_and_check(ctx, pattern):
    dfa = checked(ctx.solution.compile_regex(pattern), pattern)
    check_language(dfa, pattern)
    check_structure(dfa, pattern)
    return dfa


def canonical(dfa):
    """States renumbered breadth-first from the start, following bytes in increasing order."""
    table, start, accepting = dfa
    order = {start: 0}
    queue = deque([start])
    while queue:
        for _, target in sorted(table[queue.popleft()].items()):
            if target not in order:
                order[target] = len(order)
                queue.append(target)
    renamed = {order[s]: {b: order[t] for b, t in row.items()} for s, row in table.items()}
    return renamed, {order[s] for s in accepting}


# -- agreement with Python's re ----------------------------------------------------------

EXTRA = 'aZ09 _-.\n\t"\\é€😀'


def texts_for(rng, pattern, n):
    """Texts near the language: strings of it, corrupted copies, and random strings."""
    table, start, accepting = kit_dfa(pattern)
    alphabet = sorted({chr(b) for row in table.values() for b in row if b < 0x80} | set(EXTRA))
    out = [""]
    for _ in range(n):
        state, data = start, bytearray()
        for _ in range(int(rng.integers(0, 14))):  # a random walk through the automaton
            if not table[state] or (state in accepting and rng.random() < 0.2):
                break
            byte = int(rng.choice(sorted(table[state])))
            data.append(byte)
            state = table[state][byte]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out.append(text)
        if text:  # corrupt it: drop, replace, or insert one character
            at = int(rng.integers(0, len(text)))
            other = str(rng.choice(alphabet))
            out += [text[:at] + text[at + 1 :], text[:at] + other + text[at + 1 :],
                    text[:at] + other + text[at:], text + other]  # fmt: skip
        out.append("".join(rng.choice(alphabet, size=int(rng.integers(1, 6)))))
    return out


def check_against_re(ctx, dfa, pattern, n=12):
    compiled = re.compile(pattern, re.ASCII)
    for text in texts_for(ctx.rng, pattern, n):
        want = compiled.fullmatch(text) is not None
        got = accepts(dfa, text.encode("utf-8"))
        if got != want:
            verdict = "accepts" if got else "rejects"
            check.that(
                False,
                f"{FN}: the automaton {verdict} a text on which re.fullmatch says otherwise",
                f"pattern: {pattern!r}\ntext: {text!r}\nre.fullmatch(pattern, text, re.ASCII) "
                f"{'matches' if want else 'does not match'}",
            )


def check_patterns(ctx, patterns, with_re=True):
    for pattern in patterns:
        dfa = compile_and_check(ctx, pattern)
        if with_re:
            check_against_re(ctx, dfa, pattern)


# -- cases ------------------------------------------------------------------------------

INTEGER = r"-?(0|[1-9]\d*)"
DIGITS, NONZERO = range(0x30, 0x3A), range(0x31, 0x3A)


@case("the worked example: a JSON integer, -?(0|[1-9]\\d*), in four states", public=True)
def worked_example(ctx):
    dfa = compile_and_check(ctx, INTEGER)
    want = {
        0: {0x2D: 1, 0x30: 2, **dict.fromkeys(NONZERO, 3)},
        1: {0x30: 2, **dict.fromkeys(NONZERO, 3)},
        2: {},
        3: dict.fromkeys(DIGITS, 3),
    }
    table, accepting = canonical(dfa)
    check.equal(len(table), 4, label="number of states")
    check.that(
        table == want and accepting == {2, 3},
        f"{FN}({INTEGER!r}): the automaton is not the one of the statement",
        f"states renumbered breadth-first from the start:\n  transitions = {table}\n"
        f"  accepting = {sorted(accepting)}\nexpected:\n  transitions = {want}\n"
        "  accepting = [2, 3]",
    )
    for text, verdict in (("0", True), ("-7", True), ("120", True), ("", False), ("-", False),
                          ("007", False), ("-0", True), ("1a", False)):  # fmt: skip
        check.that(
            accepts(dfa, text.encode()) == verdict,
            f"{FN}({INTEGER!r}): the automaton {'rejects' if verdict else 'accepts'} {text!r}, "
            f"expected it to be {'accepted' if verdict else 'rejected'}",
        )


# Rejects alternation_binds_tighter ("ab|cd" read as a(b|c)d) and star_binds_sequence.
@case("precedence: repetition binds tighter than concatenation, alternation loosest",
      public=True)  # fmt: skip
def precedence(ctx):
    check_patterns(ctx, ["abc", "ab|cd", "a|b|c", "a(b|c)d", "(ab|cd)e", "ab*", "ab*c|d",
                         "a|bc+", "(?:ab)+|c", "x(a|bc)*y"])  # fmt: skip
    dfa = compile_and_check(ctx, "ab|cd")
    for text, verdict in (("ab", True), ("cd", True), ("abd", False), ("acd", False)):
        check.that(
            accepts(dfa, text.encode()) == verdict,
            f"{FN}('ab|cd'): the automaton {'rejects' if verdict else 'accepts'} {text!r}, "
            f"expected it to be {'accepted' if verdict else 'rejected'}",
            "the branches are ab and cd",
        )


# Rejects upper_bound_off_by_one, open_bound_is_exact, and plus_allows_zero.
@case("quantifiers: * + ? {m} {m,} {m,n}, on characters and on groups", public=True)
def quantifiers(ctx):
    check_patterns(ctx, ["a*", "a+", "a?", "ab?c", "(ab)*", "a{3}", "a{2,4}", "a{2,}", "a{0}",
                         "a{0,1}", "a{1,1}", "(a|b){2,3}c", "(ab){1,2}", "x{0,}y", "(a{2}){2}",
                         "[ab]{3,}c"])  # fmt: skip


# Rejects range_excludes_end and dot_matches_newline.
@case("classes, ranges, negation, and escapes", public=True)
def classes_and_escapes(ctx):
    check_patterns(ctx, ["[abc]", "[a-f0-9]+", "[^abc]", r"\d+", r"\w+", r"\s*x", r"[\d_]+",
                         r"[\]\\\-\^]+", r"\.\*\+\?\(\)\[\]\{\}\|\\", r"\x41\x7e", r"a\n\t\r\f\v\0b",
                         r"[\n\t ]*", r"[^\n\"]*", r".", r"a.c", r"[\x30-\x39a]", r"[a\-z]",
                         r"\"[^\"\\]*\"", r"[+\-]?\d", r"[a-a]", r"\/\-\_\ "])  # fmt: skip


# Rejects negated_class_any_byte: é is two bytes and one character.
@case("bytes: ., [^...], \\D \\W \\S match one whole UTF-8 character, never half of one",
      public=True)  # fmt: skip
def whole_characters(ctx):
    for pattern, good, bad in (
        (
            ".",
            ["a", "é", "€", "😀"],
            [
                b"",
                b"\n",
                b"\xc3",
                b"\xa9",
                b"\x80",
                b"\xc3\xa9\xa9",
                "éa".encode(),
                b"\xe2\x82",
                b"\xff",
                b"\xc1\x81",
            ],
        ),
        (r'[^"]*', ["", "né €", "a\nb", "😀😀"], [b'"', b"\xe2\x82", b"a\x9f", b"\xf0\x9f\x98"]),
        (r"\D", ["x", "é", "\n"], [b"7", b"\xc3"]),
        (r"\W\S", ["-é", "é€", " 😀"], [b"a-", b"\xc3\xa9", b"- "]),
    ):
        dfa = compile_and_check(ctx, pattern)
        for text in good:
            check.that(
                accepts(dfa, text.encode("utf-8")),
                f"{FN}({pattern!r}): the automaton rejects a text the pattern matches",
                f"text: {text!r}, bytes {text.encode('utf-8')!r}",
            )
        for data in bad:
            check.that(
                not accepts(dfa, data),
                f"{FN}({pattern!r}): the automaton accepts bytes the pattern does not match",
                f"bytes: {data!r}",
            )
        check_against_re(ctx, dfa, pattern)


# Rejects no_minimization and explicit_dead_state.
@case("minimal: equivalent states are merged and no dead state is kept", public=True)
def minimal(ctx):
    for pattern, states in (("a*a*", 1), ("(a|aa)*", 1), ("ab|ac", 3), ("(a|b)(a|b)", 3),
                            ("(a|b)*abb", 4), ("x(a|b)*|x(a|b)*c?", 3), ("a{2,4}", 5),
                            ("(ab|cb)+", 3), ("", 1), ("a|", 2), ("[0-4]|[5-9]", 2)):  # fmt: skip
        dfa = compile_and_check(ctx, pattern)
        check.equal(len(dfa[0]), states, label=f"number of states for {pattern!r}")


INVALID = [
    "(", "a)", "(a", "(a|b", "())", "[a", "]", "}", "{", "a{2", "*a", "+", "a|*", "(?)", "(*a)",
    "a**", "a*?", "a+?", "a??", "a*+", "a{2}{3}", "a{1,2}?", "a{}", "a{,3}", "a{3,2}", "a{x}",
    "a{1,x}", "a{101}", "a{2,101}", "^a", "a$", "a|^b", "(?=a)", "(?!a)", "(?P<n>a)", "(?i)a",
    "(?<=a)b", r"\b", r"\A", r"\Z", r"\1", r"\p", r"\u0041", r"\q", r"\x4", r"\xg1", "a\\",
    "[z-a]", r"[a-\d]", r"[\D]", r"[\W]", r"[^\S]", "[9-0]",
]  # fmt: skip


def check_invalid(ctx, patterns):
    for pattern in patterns:
        try:
            ctx.solution.compile_regex(pattern)
        except ValueError:
            continue
        check.that(
            False,
            f"{FN}: returned an automaton for an invalid pattern instead of raising ValueError",
            f"pattern: {pattern!r}",
        )


# Rejects accepts_lazy.
@case("invalid patterns raise ValueError: unbalanced, nothing to repeat, lazy, anchors",
      public=True)  # fmt: skip
def invalid_patterns(ctx):
    check_invalid(ctx, ["(", "a)", "[a", "*a", "a**", "a*?", "a{2", "a{3,2}", "^a", "a$",
                        "(?=a)", r"\b", r"\1", "[z-a]", r"[\D]"])  # fmt: skip


@case("every invalid pattern the statement lists")
def invalid_patterns_all(ctx):
    check_invalid(ctx, INVALID)


@case("empty branches and nested repetition: |a, a||b, (a*)*, (a|)+, (a?){3}, ()")
def empty_and_nested(ctx):
    check_patterns(ctx, ["|a", "a||b", "(a*)*", "(a|)+", "(a?){3}", "()", "()*", "a()b", "(|)",
                         "(a*b*)*", "(a+)+", "((a))", "(a|b|)c", "(?:)x", "(a{0}){5}b"])  # fmt: skip


@case("patterns from real constraints: date, time, number, string, identifier, IPv4")
def realistic(ctx):
    check_patterns(ctx, [
        r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])",
        r"([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?",
        r"-?(0|[1-9]\d*)(\.\d+)?([eE][+\-]?\d+)?",
        r'"([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"',
        r"[A-Za-z_]\w*",
        r"(\d{1,3}\.){3}\d{1,3}",
        r'\{"name":"[a-z]+","age":\d{1,3}\}',
        r"(true|false|null)",
        r"\[(\d+(,\d+)*)?\]",
        r"[\w.+\-]+@[\w\-]+(\.[\w\-]+)+",
        r"(yes|no|maybe)( (yes|no|maybe)){0,3}",
        r"#[0-9a-fA-F]{6}",
    ])  # fmt: skip


@case("large bounds and wide classes: [a-z]{0,100}, .{0,12}, \\w{8,}, (ab){50}")
def large_bounds(ctx):
    check_patterns(ctx, ["[a-z]{0,100}", ".{0,12}", r"\w{8,}", "(ab){50}", "a{100}",
                         r"[^,]{1,20},", "(a|b){0,30}c"], with_re=False)  # fmt: skip
    dfa = compile_and_check(ctx, "[a-z]{0,100}")
    check.equal(len(dfa[0]), 101, label="number of states")


@case("a language whose minimal automaton is large: (a|b)*a(a|b){6} needs 128 states")
def exponential(ctx):
    dfa = compile_and_check(ctx, "(a|b)*a(a|b){6}")
    check.equal(len(dfa[0]), 128, label="number of states")
    check_against_re(ctx, dfa, "(a|b)*a(a|b){6}", n=30)


ATOMS = [*"abc01 -_", ".", r"\d", r"\w", r"\s", r"\D", r"\.", r"\n", r"\x61", r"\|",
         "[ab]", "[a-c]", "[0-9]", "[a-c_]", r"[\d_]", r"[\s0]", r"[\-a]", "[ac-e]",
         "[^ab]", "[^a-c]", r"[^\d]", r"[^\n ]"]  # fmt: skip
QUANTIFIERS = ["*", "+", "?", "{2}", "{1,3}", "{2,}", "{0,2}"]


def random_pattern(rng, depth=0):
    """A random valid pattern of the subset, as a string."""
    kinds = ["atom", "atom", "rep"] + ["alt", "cat", "cat", "rep"] * (depth < 3)
    kind = str(rng.choice(kinds))
    if kind == "atom":
        return str(rng.choice(ATOMS))
    if kind == "alt":
        parts = [random_pattern(rng, depth + 1) for _ in range(int(rng.integers(2, 4)))]
        return "(" + "|".join(parts) + ")"
    if kind == "cat":
        return "".join(random_pattern(rng, depth + 1) for _ in range(int(rng.integers(2, 4))))
    # A quantifier applies to one atom or to a group.
    if depth < 3 and rng.random() < 0.6:
        inner = "(" + ("?:" * int(rng.integers(0, 2))) + random_pattern(rng, depth + 1) + ")"
    else:
        inner = str(rng.choice(ATOMS))
    return inner + str(rng.choice(QUANTIFIERS))


@case("random patterns of the subset: equal to the minimal automaton, and to re on texts")
def random_patterns(ctx):
    def prop(rng):
        pattern = random_pattern(rng)
        re.compile(pattern)  # the generator only writes valid patterns
        dfa = compile_and_check(ctx, pattern)
        check_against_re(ctx, dfa, pattern, n=8)

    check.forall(prop, trials=40)
