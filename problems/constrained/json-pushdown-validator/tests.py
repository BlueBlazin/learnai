"""Tests for json-pushdown-validator.

Three sources of truth are used:

* hand-written fixtures: valid texts, unfinished texts, and near-misses with the index
  of the first character that must be rejected;
* the oracle reference, for the verdict on every possible next character at every
  prefix (which also proves that old states stay usable);
* Python's `json` module, as an independent check of `json_complete` on whole texts.

States are opaque and must not be modified by `json_step`, so the functions are called
directly on the solution module: `ctx.call` would copy the state on every call and
hide a solution that mutates it.
"""

import json

from whetstone.judge import case, check

# Characters tried as "the next character" at every prefix: structure, whitespace,
# digits and number punctuation, the letters of the literals, string specials, a
# control character, DEL, and non-ASCII.
ALPHABET = "{}[]:,\" \n\t\r0159-+.eEtrufalsn\\/bxAZ\x01\x1f\x7fé😀'_"

# (text, index of the first character that json_step must reject)
NEAR_MISSES = [
    # trailing and stray commas
    ("[1,]", 3), ('{"a":1,}', 7), ("[,1]", 1), ("{,}", 1), ("[1,,2]", 3), ('{"a":"b",}', 9),
    ('{"a":1,,"b":2}', 7),
    # numbers
    ("01", 1), ("-01", 2), ("[00]", 2), ("00", 1), ("-00", 2), ('{"a":01}', 6), ("1.e5", 2),
    (".5", 0), ("[.1]", 1), ("[1.]", 3), ("[0.]", 3), ("+1", 0), ("--1", 1), ("0x10", 1),
    ("[1.5.2]", 4), ("[1e5e2]", 4), ("[-]", 2), ("[1-2]", 2), ("-a", 1), ("1ee", 2),
    ("1_000", 1), ("0.e1", 2), ("NaN", 0), ("Infinity", 0), ("-Infinity", 1),
    # brackets
    ("[1}", 2), ('{"a":1]', 6), ("[]]", 2), ("{}}", 2), ("[1]]", 3), ("]", 0), ("}", 0),
    ('[{"a":[1}]', 8), ('{"a":[1,2}', 9), ("[[1],[2}}", 7),
    # separators and keys
    ("1 2", 2), ("[1 2]", 3), ("[0 1]", 3), ("0 0", 2), ('{"a" 1}', 5), ('{"a":}', 5),
    ('{"a"::1}', 5), ("{a:1}", 1), ("{1:2}", 1), ('{"a":1 "b":2}', 7), ("[1;2]", 2),
    ('{"a":1;"b":2}', 6), (":", 0), (",", 0), ("[:]", 1), ("{[]:1}", 1), ('{"a":1}x', 7),
    ("{}{}", 2), ("[] []", 3), ('"a" "b"', 4), ('["a":1]', 4), ('{"a","b"}', 4),
    # strings
    ("'a'", 0), ('"a\nb"', 2), ('"a\tb"', 2), ('"\x00"', 1), ('"\x1f"', 1), ('"\\x41"', 2),
    ('"\\a"', 2), ("\"\\'\"", 2), ('"\\u12G4"', 5), ('"\\u12"', 5), ('"\\U0041"', 2),
    ('{"k\n":1}', 3), ('"\\u+123"', 3),
    # literals
    ("tRue", 1), ("True", 0), ("nulll", 4), ("truefalse", 4), ("[true,fals]", 10), ("[tru]", 4),
    ("nil", 1), ("[false1]", 6),
    # other whitespace, byte order mark, comments
    ("\ufeff[]", 0), ("\x0b1", 0), ("\x0c[]", 0), ("\xa0[]", 0), ("//", 0), ("/**/1", 0),
    ("[1,/*x*/2]", 3),
]  # fmt: skip

# Valid as far as they go, but not complete documents.
UNFINISHED = [
    "", " ", "\n\t ", "-", "1.", "0.", "1e", "1e+", "2.5E", "0e", "1E-", "t", "tru", "nul", "fals",
    '"', '"abc', '"\\', '"\\u', '"\\u12', "[", "[1", "[1,", "[1, ", "{", '{"a"', '{"a":', '{"a":1',
    '{"a":1,', '{"a":1, ', "[[[", '[{"a":[', '{"a":{"b":[1,{"c":"', "[1.", "[-", "[1e", '["\\',
]  # fmt: skip

VALID = [
    "0", "-0", "-0.0", "7", "1E+2", "1e-2", "12.50", "1.0e+10", "0e0", "0E-0", "123456789012345678",
    "true", "false", "null", '"a"', '""', "[]", "{}", "[ ]", "{ }", " [1 , 2 ] ", "\t\n\r true \n",
    '{"":0}', '{"a":1,"a":2}', "[[[[]]]]", '{"a":{"b":{"c":[{}]}}}', "[0]", '{"a":[],"b":{}}',
    '[1e5,-2,0.5,"x",null,true,false,{},[]]', '{"a": [1, 2.5e3, true], "b": null}',
    '"\\u00e9\\/\\b\\f\\n\\r\\t\\"\\\\"', '"\x7f é€😀 \u2028"', '"\\uAbCd\\u0000"',
    '"\\ud800"', '"\\uDE00\\uD83D"', '"\\uD83D\\uDE00"', '["\\udfff", "\\uD800x"]',
    '{"\\ud83d": "\\ude00"}', '"{[,:]}"', '"tru"', '"//"', '[-0, 0.0, 0e1, 1E1]',
    '{ "k" : [ ] , "m" : { } }', '[\n  1,\n  2\n]\n',
]  # fmt: skip


def python_accepts(text):
    """Whether Python's json module, with its two extensions disabled, accepts `text`."""

    def reject_constant(name):  # NaN, Infinity, -Infinity are not JSON
        raise ValueError(name)

    try:
        json.loads(text, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return False
    return True


def feed(module, text, state=None):
    """Step through `text`. Returns (state, None), or (None, i) if character i is rejected."""
    if state is None:
        state = module.json_start()
    for i, ch in enumerate(text):
        try:
            state = module.json_step(state, ch)
        except module.Invalid:
            return None, i
    return state, None


def complete(module, state, text):
    got = module.json_complete(state)
    check.that(
        got is True or got is False,
        "json_complete must return True or False",
        f"after {text!r} it returned {got!r}",
    )
    return got


def expect_prefix(ctx, text, is_complete):
    """Every character of `text` is accepted, and json_complete agrees with `is_complete`."""
    state, bad = feed(ctx.solution, text)
    check.that(
        bad is None,
        "json_step raised Invalid inside a text that is a prefix of valid JSON",
        f"text: {text!r}\nrejected character: index {bad}"
        + (f", {text[bad]!r}\n" if bad is not None else "\n"),
    )
    got = complete(ctx.solution, state, text)
    check.that(
        got == is_complete,
        "json_complete is wrong at the end of a text",
        f"text: {text!r}\njson_complete returned {got}, expected {is_complete}\n",
    )


def expect_invalid_at(ctx, text, index):
    """The first rejected character of `text` is the one at `index`."""
    _, bad = feed(ctx.solution, text)
    check.that(
        bad is not None,
        "json_step accepted a text that cannot become valid JSON",
        f"text: {text!r}\nthe character at index {index} ({text[index]!r}) must raise Invalid\n",
    )
    check.that(
        bad == index,
        "json_step raised Invalid at the wrong character",
        f"text: {text!r}\nexpected at index {index} ({text[index]!r}), "
        f"raised at index {bad} ({text[bad]!r})\n",
    )


def same_frontier(ctx, text):
    """At every prefix of `text`: the same next characters are accepted as by the oracle,
    and json_complete agrees. The solution's states are stepped many times each."""
    sol, ref = ctx.solution, ctx.oracle
    state, ref_state = sol.json_start(), ref.json_start()
    for i in range(len(text) + 1):
        prefix = text[:i]
        got = complete(sol, state, prefix)
        expected = ref.json_complete(ref_state)
        check.that(
            got == expected,
            "json_complete is wrong",
            f"text so far: {prefix!r}\njson_complete returned {got}, expected {expected}\n",
        )
        for ch in ALPHABET:
            _, ref_bad = feed(ref, ch, ref_state)
            _, bad = feed(sol, ch, state)
            if (bad is None) != (ref_bad is None):
                verdict = "accepted" if bad is None else "rejected"
                wanted = "raise Invalid" if bad is None else "be accepted"
                check.that(
                    False,
                    f"json_step {verdict} a character it must not",
                    f"text so far: {prefix!r}\nnext character: {ch!r} must {wanted}\n"
                    "(the state for this prefix had been stepped with other characters "
                    "before; json_step must not modify its argument)\n",
                )
        if i == len(text):
            break
        _, ref_bad = feed(ref, text[i], ref_state)
        if ref_bad is not None:
            break
        ref_state = ref.json_step(ref_state, text[i])
        state = sol.json_step(state, text[i])


# -- random documents ---------------------------------------------------------------


def random_ws(rng):
    return (
        "".join(rng.choice(list(" \n\t\r"), size=int(rng.integers(0, 3))))
        if rng.random() < 0.3
        else ""
    )


def random_string(rng):
    pieces = []
    for _ in range(int(rng.integers(0, 6))):
        kind = rng.random()
        if kind < 0.5:
            pieces.append(str(rng.choice(list("abz 09_{}[]:,/é€😀\x7f"))))
        elif kind < 0.7:
            pieces.append("\\" + str(rng.choice(list('"\\/bfnrt'))))
        else:
            code = int(rng.choice([0x41, 0xE9, 0x0, 0x1F, 0xD800, 0xDBFF, 0xDC00, 0xDFFF, 0xFFFF]))
            digits = f"{code:04x}"
            pieces.append("\\u" + (digits.upper() if rng.random() < 0.5 else digits))
    return '"' + "".join(pieces) + '"'


def random_number(rng):
    text = "-" if rng.random() < 0.3 else ""
    text += "0" if rng.random() < 0.3 else str(int(rng.integers(1, 10000)))
    if rng.random() < 0.4:
        text += "." + "".join(
            str(int(d)) for d in rng.integers(0, 10, size=int(rng.integers(1, 4)))
        )
    if rng.random() < 0.3:
        text += str(rng.choice(["e", "E"])) + str(rng.choice(["", "+", "-"]))
        text += str(int(rng.integers(0, 400)))
    return text


def random_value(rng, depth):
    kind = rng.random()
    if depth <= 0 or kind < 0.45:
        scalar = rng.random()
        if scalar < 0.35:
            return random_number(rng)
        if scalar < 0.7:
            return random_string(rng)
        return str(rng.choice(["true", "false", "null"]))
    n = int(rng.integers(0, 4))
    if kind < 0.72:
        items = [random_ws(rng) + random_value(rng, depth - 1) + random_ws(rng) for _ in range(n)]
        return "[" + (",".join(items) if items else random_ws(rng)) + "]"
    members = [
        random_ws(rng) + random_string(rng) + random_ws(rng) + ":"
        + random_ws(rng) + random_value(rng, depth - 1) + random_ws(rng)
        for _ in range(n)
    ]  # fmt: skip
    return "{" + (",".join(members) if members else random_ws(rng)) + "}"


def random_document(rng):
    text = random_ws(rng) + random_value(rng, int(rng.integers(0, 5))) + random_ws(rng)
    if not python_accepts(text):  # an authoring error, not a solution failure
        raise RuntimeError(f"the generator produced invalid JSON: {text!r}")
    return text


# -- cases ---------------------------------------------------------------------------


@case("the worked example: an object, character by character", public=True)
def worked_example(ctx):
    sol = ctx.solution
    check.that(
        isinstance(sol.Invalid, type) and issubclass(sol.Invalid, Exception),
        "Invalid must be an exception class",
    )
    text = '{"a": [1, 2.5e3, true], "b": null}'
    state = sol.json_start()
    check.that(not complete(sol, state, ""), "json_complete is True before any character")
    for i, ch in enumerate(text):
        try:
            state = sol.json_step(state, ch)
        except sol.Invalid:
            check.that(
                False,
                "json_step raised Invalid inside the worked example",
                f"text: {text!r}\nrejected character: index {i}, {ch!r}\n",
            )
        done = complete(sol, state, text[: i + 1])
        check.that(
            done == (i == len(text) - 1),
            "json_complete is wrong inside the worked example",
            f"after {text[: i + 1]!r} it returned {done}\n",
        )
    after_space = sol.json_step(state, " ")
    check.that(complete(sol, after_space, text + " "), "trailing whitespace keeps a text complete")
    for label, start in (("the complete object", state), ("the trailing space", after_space)):
        check.raises(
            sol.Invalid, sol.json_step, start, "x", label=f"json_step(…, 'x') after {label}"
        )
    expect_invalid_at(ctx, '{"a": [1, 2.5e3, true}', 21)
    expect_invalid_at(ctx, '{"a": [1, 02]}', 11)


@case("scalars at the top level, and when a text counts as complete", public=True)
def scalars_and_completeness(ctx):
    # Rejects complete_before_value and complete_mid_number.
    for text in ["0", "-1", "12", "1.5", "1e5", "true", '"x"', " null ", "\n7\n"]:
        expect_prefix(ctx, text, True)
    for text in ["", "  ", "-", "1.", "1e", "1e+", "tru", '"x', "[1", '{"a":1']:
        expect_prefix(ctx, text, False)
    # A number is complete as soon as it could end, and stays extendable.
    expect_prefix(ctx, "1", True)
    expect_prefix(ctx, "12", True)
    expect_prefix(ctx, "12.", False)
    expect_prefix(ctx, "12.5", True)
    expect_prefix(ctx, "12.5e", False)
    expect_prefix(ctx, "12.5e-", False)
    expect_prefix(ctx, "12.5e-3", True)


@case("near-misses: trailing commas, leading zeros, mismatched brackets", public=True)
def common_near_misses(ctx):
    # Rejects trailing_comma, leading_zeros, and unmatched_closers.
    for text, index in [
        ("[1,]", 3), ('{"a":1,}', 7), ("01", 1), ("-01", 2), ('{"a":01}', 6), ("[1}", 2),
        ('{"a":1]', 6), ('[{"a":[1}]', 8), ("[1 2]", 3), ("{a:1}", 1), ("1.e5", 2), ("[]]", 2),
    ]:  # fmt: skip
        expect_invalid_at(ctx, text, index)
    for text in ["[1,2]", '{"a":1,"b":2}', "10", "-10.01", "[0,0]", '{"a":[{}]}', "[[],[]]"]:
        expect_prefix(ctx, text, True)


@case("a state stays usable after it has been stepped", public=True)
def states_are_values(ctx):
    # Rejects mutates_stack: one saved state is continued in several ways.
    sol = ctx.solution
    saved, _ = feed(sol, '[{"a":1}')
    check.that(saved is not None, 'json_step raised Invalid inside [{"a":1}')
    closed = sol.json_step(saved, "]")
    check.that(complete(sol, closed, '[{"a":1}]'), 'json_complete is False after [{"a":1}]')
    # Closing the array through `closed` must not have changed `saved`.
    for tail, is_complete in [(",2]", True), ("]", True), (",[", False), (' , {"b":[]} ] ', True)]:
        state, bad = feed(sol, tail, saved)
        check.that(
            bad is None,
            "a saved state behaves differently after another state was derived from it",
            f'saved state: after [{{"a":1}}\nthen stepped with "]" (result kept separately)\n'
            f"then the saved state was continued with {tail!r}: Invalid at index {bad}\n",
        )
        check.that(
            complete(sol, state, tail) == is_complete,
            "json_complete is wrong for a continuation of a saved state",
            f'saved state: after [{{"a":1}}, continued with {tail!r}\n',
        )
    check.raises(sol.Invalid, sol.json_step, saved, "}", label='json_step(saved, "}")')
    same_frontier(ctx, '{"k": [1, {"m": []}], "n": "x"}')


@case("valid documents: every prefix is accepted and only the whole text is complete")
def valid_documents(ctx):
    for text in VALID:
        expect_prefix(ctx, text, True)
        if not python_accepts(text):  # an authoring error
            raise RuntimeError(f"fixture is not valid JSON: {text!r}")
    for text in UNFINISHED:
        expect_prefix(ctx, text, False)


@case("strings: escapes, any four hex digits after \\u, raw control characters")
def strings(ctx):
    # Rejects raw_control_chars and rejects_surrogates.
    for code in range(0x20):
        expect_invalid_at(ctx, f'"a{chr(code)}"', 2)
        expect_invalid_at(ctx, f'{{"{chr(code)}":1}}', 2)
        expect_prefix(ctx, f'"\\u{code:04x}"', True)
        if chr(code) not in " \t\n\r":
            expect_invalid_at(ctx, f"[1,{chr(code)}2]", 3)
    for text in ['"\x7f"', '"\x80\xa0"', '"\u2028\u2029"', '"\U0010ffff"', '"\\u0000"']:
        expect_prefix(ctx, text, True)
    for escape in '"\\/bfnrt':
        expect_prefix(ctx, f'"\\{escape}"', True)
    for escape in "aAeNRTU0x'vz ":
        expect_invalid_at(ctx, f'"\\{escape}"', 2)
    for digits in ["d800", "DBFF", "dc00", "DFFF", "D83D", "de00", "dAbC", "ffff", "0000", "FfFf"]:
        expect_prefix(ctx, f'"\\u{digits}"', True)
        expect_prefix(ctx, f'{{"\\u{digits}x": "\\u{digits}\\u{digits}"}}', True)
    for text, index in [('"\\u12g4"', 5), ('"\\u"', 3), ('"\\u1"', 4), ('"\\u123"', 6)]:
        expect_invalid_at(ctx, text, index)
    same_frontier(ctx, '["\\ud83d\\ude00\\\\", "\\"", "\\u00E9"]')


@case("numbers: every way to write one, and every way to get one wrong")
def numbers(ctx):
    for number in ["0", "-0", "5", "-5", "10", "0.0", "0.5", "10.05", "0e0", "0E5", "1e+5", "1E-5",
                   "-1.5e10", "9007199254740993", "1.000", "100e001"]:  # fmt: skip
        expect_prefix(ctx, number, True)
        expect_prefix(ctx, f"[{number}]", True)
        expect_prefix(ctx, f"[{number},{number} , {number}\n]", True)
        expect_prefix(ctx, f'{{"n":{number}}}', True)
        expect_prefix(ctx, f" {number} ", True)
        expect_invalid_at(ctx, f"{number} {number}", len(number) + 1)
    for bad in ["00", "01", "-00", "-01", "0123", "-0123"]:
        index = bad.index("0") + 1
        expect_invalid_at(ctx, bad, index)
        expect_invalid_at(ctx, f"[{bad}]", index + 1)
        expect_invalid_at(ctx, f'{{"n": {bad}}}', index + 6)
    same_frontier(ctx, "[-0.5e+10, 0, 12E3]")
    same_frontier(ctx, "-12.75e-3")


@case("the full list of near-misses, each rejected at exactly the right character")
def all_near_misses(ctx):
    for text, index in NEAR_MISSES:
        if python_accepts(text):  # an authoring error
            raise RuntimeError(f"fixture is valid JSON: {text!r}")
        expect_invalid_at(ctx, text, index)


@case("deep nesting: 600 containers opened, then closed in the right and the wrong order")
def deep_nesting(ctx):
    sol = ctx.solution
    kinds = [("[", "]") if ctx.rng.random() < 0.5 else ('{"k":', "}") for _ in range(600)]
    opening = "".join(open_ for open_, _ in kinds)
    closing = "".join(close for _, close in reversed(kinds))
    expect_prefix(ctx, opening, False)
    expect_prefix(ctx, opening + "0" + closing, True)
    expect_prefix(ctx, opening + "0" + closing[:-1], False)
    # Swap one closer for the other kind, somewhere in the middle.
    position = 300
    wrong = "}" if closing[position] == "]" else "]"
    text = opening + "0" + closing[:position] + wrong + closing[position + 1 :]
    expect_invalid_at(ctx, text, len(opening) + 1 + position)
    state, _ = feed(sol, opening + "0" + closing)
    check.raises(sol.Invalid, sol.json_step, state, "]", label="one closing bracket too many")


@case("property: every prefix of a random document is accepted; the frontier matches")
def random_documents(ctx):
    def prop(rng):
        text = random_document(rng)
        sol = ctx.solution
        state = sol.json_start()
        for i, ch in enumerate(text):
            try:
                state = sol.json_step(state, ch)
            except sol.Invalid:
                check.that(
                    False,
                    "json_step raised Invalid inside a valid JSON text",
                    f"text: {text!r}\nrejected character: index {i}, {ch!r}\n",
                )
            prefix = text[: i + 1]
            got = complete(sol, state, prefix)
            check.that(
                got == python_accepts(prefix),
                "json_complete disagrees with a JSON parser on a prefix of a valid text",
                f"text so far: {prefix!r}\njson_complete returned {got}\n",
            )
        same_frontier(ctx, text[:60])

    check.forall(prop, trials=40)


@case("property: a random edit of a valid document is rejected at the right character, or not")
def random_edits(ctx):
    def prop(rng):
        text = random_document(rng)
        for _ in range(8):
            i = int(rng.integers(0, len(text) + 1))
            ch = str(rng.choice(list(ALPHABET)))
            kind = rng.random()
            if kind < 0.3:
                edited = text[:i] + text[i + 1 :]
            elif kind < 0.6:
                edited = text[:i] + ch + text[i:]
            elif kind < 0.9:
                edited = text[:i] + ch + text[i + 1 :]
            else:
                edited = text[:i]
            ref_state, ref_bad = feed(ctx.oracle, edited)
            ref_complete = ref_bad is None and ctx.oracle.json_complete(ref_state)
            if ref_complete != python_accepts(edited):  # an authoring error
                raise RuntimeError(f"the oracle and the json module disagree on {edited!r}")
            state, bad = feed(ctx.solution, edited)
            detail = f"text: {edited!r}\n"
            if ref_bad is None:
                check.that(
                    bad is None,
                    "json_step raised Invalid inside a text that is a prefix of valid JSON",
                    detail + f"rejected character: index {bad}\n",
                )
                got = complete(ctx.solution, state, edited)
                check.that(
                    got == ref_complete,
                    "json_complete is wrong at the end of a text",
                    detail + f"json_complete returned {got}\n",
                )
            else:
                check.that(
                    bad is not None,
                    "json_step accepted a text that cannot become valid JSON",
                    detail + f"the character at index {ref_bad} must raise Invalid\n",
                )
                check.that(
                    bad == ref_bad,
                    "json_step raised Invalid at the wrong character",
                    detail + f"expected at index {ref_bad}, raised at index {bad}\n",
                )

    check.forall(prop, trials=40)
