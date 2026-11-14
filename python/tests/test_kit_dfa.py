"""CharDFA (hand-built and from a regex), TokenConstraint, DFAConstraint."""

import random
import re

import pytest

from whetstone.kit import DEAD, CharDFA, DFAConstraint, TokenConstraint, ToyVocab

# -- a hand-built automaton -------------------------------------------------------------

# Language: "ab" then any number of "c". State 3 is a trap: nothing accepts from it.
HAND = {0: {"a": 1, "x": 3}, 1: {b"b": 2}, 2: {ord("c"): 2}, 3: {"x": 3}}


def hand():
    return CharDFA(HAND, start=0, accepting=[2])


def test_step_walk_and_acceptance():
    dfa = hand()
    assert (dfa.start, dfa.accepting, dfa.states, dfa.n_states) == (0, frozenset({2}), (0, 1, 2, 3), 4)
    assert DEAD == CharDFA.DEAD == -1
    assert dfa.step(0, ord("a")) == 1
    assert dfa.step(1, ord("a")) == DEAD
    assert dfa.step(DEAD, ord("a")) == DEAD  # dead stays dead
    assert dfa.walk(0, b"abcc") == 2 and dfa.is_accepting(2)
    assert not dfa.is_accepting(1) and not dfa.is_accepting(DEAD)
    assert dfa.walk(1, b"") == 1


def test_steps_are_counted_and_a_dead_walk_stops():
    dfa = hand()
    dfa.step(0, ord("a"))
    dfa.step(DEAD, ord("a"))
    assert dfa.counters.steps == 2
    dfa.counters.reset()
    assert dfa.walk(0, b"abcc") == 2 and dfa.counters.steps == 4
    dfa.counters.reset()
    assert dfa.walk(0, b"azzzzzz") == DEAD
    assert dfa.counters.steps == 2  # "a", then the byte that killed the walk; the rest is not read
    dfa.counters.reset()
    assert dfa.walk(DEAD, b"abc") == DEAD and dfa.walk(0, b"") == 0
    assert dfa.counters.steps == 0


def test_oracles_are_not_counted():
    dfa = hand()
    assert dfa.accepts(b"ab") and dfa.accepts("abccc") and not dfa.accepts("a")
    assert not dfa.accepts("") and not dfa.accepts("abx")
    assert dfa.accepts_prefix("") and dfa.accepts_prefix("a") and dfa.accepts_prefix("abc")
    assert not dfa.accepts_prefix("b")
    assert dfa.is_live(0) and dfa.is_live(2) and not dfa.is_live(DEAD)
    assert dfa.counters.steps == 0


def test_trap_states_and_trim():
    dfa = hand()
    # "x" does not die, yet no continuation is accepted: the walk is stuck in a trap.
    assert dfa.walk(0, b"xx") == 3 and not dfa.is_live(3) and not dfa.accepts_prefix("x")
    trimmed = dfa.trim()
    assert trimmed.walk(0, b"x") == DEAD
    assert trimmed.states == (0, 1, 2) and trimmed.start == 0  # numbering is kept
    for text in ("ab", "abcc", "a", "", "x", "abx"):
        assert trimmed.accepts(text) == dfa.accepts(text)
        assert trimmed.accepts_prefix(text) == dfa.accepts_prefix(text)
    assert dfa.n_states == 4  # the original is untouched
    with pytest.raises(ValueError, match="accepts nothing"):
        CharDFA({0: {"a": 0}}, 0, []).trim()


def test_construction_is_validated():
    assert CharDFA({}, 0, [0]).accepts("") and CharDFA({}, 0, [0]).states == (0,)
    assert CharDFA({0: {"a": 5}}, 0, [5]).states == (0, 5)  # states mentioned anywhere exist
    for bad in ({0: {"ab": 1}}, {0: {256: 1}}, {0: {"é": 1}}, {0: {"a": -1}}, {-1: {}}, {0: {b"": 1}}):
        with pytest.raises(ValueError):
            CharDFA(bad, 0, [0])
    with pytest.raises(ValueError):
        CharDFA({}, True, [0])
    dfa = hand()
    with pytest.raises(ValueError, match="unknown DFA state"):
        dfa.step(9, ord("a"))
    for bad in (256, -1, 1.5):
        with pytest.raises(ValueError, match="byte"):
            dfa.step(0, bad)


def test_with_framework_gives_fresh_counters(fw):
    dfa = hand()
    dfa.step(0, 97)
    twin = fw.to_framework(dfa, fresh_kit=True)
    assert twin.counters.steps == 0 and dfa.counters.steps == 1
    assert twin.walk(0, b"ab") == 2 and dfa.counters.steps == 1
    assert fw.to_framework(dfa) is dfa


# -- from_regex --------------------------------------------------------------------------

PATTERNS = [
    r"-?[0-9]+", r"(true|false|null)", r'"[^"]*"', r"a{2,4}b?", r"a{3}", r"a{2,}", r"a{0,2}b",
    r"\d+(\.\d+)?", r'\{"a": (true|false)\}', r".*", r"[^a-c]x", r"(ab|a)(c|bc)", r"é+|€",
    r"\w+\s\w*", r"[a\-z]+", r"", r"(a|b)*abb", r"\S\D\W", r"[]a]+", r"[a-]+", r"(?:ab)+|c",
    r"[\d.]+", r"a|", r"\x41\t", r"[é€a]+", r'"(\\.|[^"\\])*"', r"(a*)*b", r"\[\d{1,3}(, \d{1,3})*\]",
]  # fmt: skip
ALPHABET = 'ab cxz019-.{}[],\\":truefalsn\n\tAé€😀_'


@pytest.mark.parametrize("pattern", PATTERNS)
def test_from_regex_agrees_with_re_fullmatch(pattern):
    dfa = CharDFA.from_regex(pattern)
    reference = re.compile(pattern, re.ASCII)
    rnd = random.Random(pattern)
    for _ in range(1500):
        text = "".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(0, 8)))
        assert dfa.accepts(text) == bool(reference.fullmatch(text)), text
    assert dfa.counters.steps == 0


def test_from_regex_is_minimal_trimmed_and_canonical():
    dfa = CharDFA.from_regex(r"-?[0-9]+")
    assert (dfa.start, dfa.n_states, dfa.accepting) == (0, 3, frozenset({2}))
    # Breadth-first numbering, bytes in increasing order: "-" (0x2D) before the digits.
    assert dfa.step(0, ord("-")) == 1 and dfa.step(0, ord("7")) == 2
    assert dfa.step(1, ord("7")) == 2 and dfa.step(2, ord("0")) == 2
    assert dfa.step(1, ord("-")) == DEAD and dfa.step(0, ord("a")) == DEAD
    for pattern, n_states in [(r"(true|false|null)", 11), (r"(a|b)*abb", 4), (r"", 1), (r"a{3}", 4),
                              (r"(a|a|a)(b|b)", 3), (r"x*x*x*", 1)]:  # fmt: skip
        built = CharDFA.from_regex(pattern)
        assert built.n_states == n_states, pattern
        assert all(built.is_live(state) for state in built.states)  # no trap states
    a, b = CharDFA.from_regex(r"(ab|ac)d*"), CharDFA.from_regex(r"a(b|c)d*")
    assert a._table == b._table and a.accepting == b.accepting  # same language, same automaton


def test_dead_means_not_a_prefix_for_a_regex_dfa():
    dfa = CharDFA.from_regex(r'\{"a": (true|false)\}')
    rnd = random.Random(0)
    target = '{"a": false}'
    for _ in range(500):
        text = target[: rnd.randint(0, len(target))] + rnd.choice(["", "x", "}", "t"])
        is_prefix = any(full.startswith(text) for full in ('{"a": true}', '{"a": false}'))
        assert dfa.accepts_prefix(text) == is_prefix
        assert (dfa.walk(dfa.start, text.encode()) != DEAD) == is_prefix


def test_multi_byte_characters_are_matched_whole():
    dfa = CharDFA.from_regex(r'"[^"]*"')
    euro = "€".encode()  # E2 82 AC
    state = dfa.walk(dfa.start, b'"' + euro[:2])
    assert state != DEAD and not dfa.is_accepting(state)  # alive in the middle of a character
    assert dfa.step(state, ord('"')) == DEAD  # but the character must be completed
    assert dfa.accepts('"€é😀"') and dfa.accepts_prefix(b'"\xf0\x9f')
    assert not dfa.accepts(b'"\xac"') and not dfa.accepts_prefix(b'"\xac')  # a stray continuation byte
    assert not dfa.accepts(b'"\xe2\x82"')
    dot = CharDFA.from_regex(r".")
    assert dot.accepts("é") and dot.accepts("😀") and dot.accepts("a")
    assert not dot.accepts("\n") and not dot.accepts(b"\xc3") and not dot.accepts("ab")
    literal = CharDFA.from_regex(r"é")
    assert literal.n_states == 3 and literal.walk(0, b"\xc3") == 1 and literal.accepts("é")
    raw = CharDFA.from_regex(r"\xe2\x82")
    assert raw.accepts(b"\xe2\x82")


@pytest.mark.parametrize(
    "pattern",
    [
        r"^a", r"a$", r"a*?", r"a+?", r"a??", r"a*+", r"a**", r"a{2}{3}", r"(?P<n>a)", r"(?=a)", r"(?i)a",
        r"\1", r"\b", r"\p{L}", r"a{", r"a{x}", r"a{3,2}", r"a{101}", r"{", r"}", r"]", r"*a", r"(a",
        r"a)", r"[a", r"[z-a]", r"[α-ω]", r"[^é]", r"[\D]", r"\\"[:1], r"[\xff]", r"\xg1",
    ],
)  # fmt: skip
def test_from_regex_rejects_what_it_does_not_support(pattern):
    with pytest.raises(ValueError, match="unsupported regex"):
        CharDFA.from_regex(pattern)


def test_from_regex_wants_a_string():
    with pytest.raises(TypeError):
        CharDFA.from_regex(b"a")


# -- TokenConstraint and DFAConstraint ---------------------------------------------------


def test_dfa_constraint_implements_the_protocol():
    vocab = ToyVocab()
    constraint = DFAConstraint(CharDFA.from_regex(r"(true|false)"), vocab)
    assert isinstance(constraint, TokenConstraint)
    assert not isinstance(vocab, TokenConstraint)
    state = constraint.start()
    names = lambda s: sorted(vocab.pieces[t] for t in constraint.allowed(s))
    assert names(state) == [b"f", b"fa", b"fal", b"fals", b"false", b"t", b"tr", b"tru", b"true"]
    assert constraint.allowed(state) == sorted(constraint.allowed(state))
    assert not constraint.is_accepting(state)
    state = constraint.advance(state, vocab.id_of(b"fa"))
    assert names(state) == [b"l"]
    after_l = constraint.advance(state, vocab.id_of(b"l"))
    assert names(after_l) == [b"s"]
    done = constraint.advance(constraint.advance(after_l, vocab.id_of(b"s")), vocab.id_of(b"e"))
    assert constraint.is_accepting(done) and constraint.allowed(done) == []
    assert names(state) == [b"l"]  # the old state is still usable: nothing was mutated
    assert constraint.allowed(DEAD) == []


def test_dfa_constraint_never_allows_specials_or_dying_tokens():
    vocab = ToyVocab(seed=2)
    dfa = CharDFA.from_regex(r'"[^"]*"')
    constraint = DFAConstraint(dfa, vocab)
    inside = constraint.advance(constraint.start(), vocab.id_of(b'"'))
    allowed = constraint.allowed(inside)
    assert not any(vocab.is_special(token) for token in allowed)  # empty bytes are not "allowed"
    assert vocab.id_of(b"\xe2\x82") in allowed  # half a character keeps the walk alive
    assert vocab.id_of(b"\xac") not in allowed  # a continuation byte with nothing to continue
    half = constraint.advance(inside, vocab.id_of(b"\xe2\x82"))
    # Only single continuation bytes can complete the character (any of them: the
    # automaton knows UTF-8's shape, not which code points exist).
    assert sorted(vocab.pieces[t] for t in constraint.allowed(half)) == sorted(
        bytes([b]) for b in b"\x80\x82\x98\x9f\xa9\xac"
    )
    for bad in (vocab.eos_id, vocab.id_of(b"\xac"), 999, "x"):
        with pytest.raises(ValueError, match="not allowed"):
            constraint.advance(inside, bad)
    # '"}' closes the string and then dies on "}": the whole token is refused.
    assert vocab.id_of(b'"}') not in allowed and vocab.id_of(b'"') in allowed


def test_dfa_constraint_counts_calls_but_not_dfa_steps(fw):
    vocab, dfa = ToyVocab(), CharDFA.from_regex(r"\d+")
    constraint = DFAConstraint(dfa, vocab)
    state = constraint.start()
    constraint.allowed(state)
    constraint.advance(state, vocab.id_of(b"7"))
    assert constraint.counters.snapshot() == {"allowed_calls": 1, "advance_calls": 1}
    assert dfa.counters.steps == 0 and vocab.counters.lookups == 0
    twin = fw.to_framework(constraint, fresh_kit=True)
    assert twin.counters.allowed_calls == 0 and twin.allowed(state) == constraint.allowed(state)
    assert constraint.counters.allowed_calls == 2


def test_any_allowed_sequence_is_a_prefix_of_the_language_and_can_finish():
    vocab = ToyVocab(seed=1)
    rnd = random.Random(7)
    for pattern in (r'\{"[a-z]+": (true|false|null|-?\d+(\.\d+)?)\}', r'"[^"]*"', r"\[\d+(, \d+)*\]"):
        dfa = CharDFA.from_regex(pattern)
        constraint = DFAConstraint(dfa, vocab)
        for _ in range(40):
            state, data = constraint.start(), b""
            for _ in range(12):
                allowed = constraint.allowed(state)
                if not allowed:
                    break
                token = rnd.choice(allowed)
                state = constraint.advance(state, token)
                data += vocab.pieces[token]
                assert dfa.accepts_prefix(data)
                assert constraint.is_accepting(state) == dfa.accepts(data)
            assert constraint.allowed(state) or constraint.is_accepting(state)  # never stuck


def test_dfa_constraint_respects_trap_states_of_a_hand_built_dfa():
    vocab = ToyVocab()
    # After "1" nothing is accepted, though the automaton keeps reading digits.
    dfa = CharDFA({0: {"0": 1, "1": 2}, 2: {"0": 2}}, 0, [1])
    constraint = DFAConstraint(dfa, vocab)
    assert [vocab.pieces[t] for t in constraint.allowed(0)] == [b"0"]
