"""Tests for json-schema-regex.

The verdict on a document never comes from a regular expression. `conforms` is a small
recursive matcher that reads the compact JSON text against the schema directly (it is
the statement's rules, in code), and every case compares
`re.fullmatch(schema_to_regex(schema), document)` with it on

* hand-written fixtures: documents that conform and near-misses that do not;
* generated instances of the schema, serialized compactly, which must all match;
* corrupted copies of those instances (a character dropped, doubled, or replaced, a
  space inserted, members removed or swapped), whatever verdict `conforms` gives them.

The pattern itself is checked too: it must compile with `re`, and it must stay inside
the regex subset of `regex-to-dfa`, which is tested by compiling it with the kit's
`CharDFA.from_regex` (for patterns small enough to compile quickly).
"""

import json
import re

from whetstone.judge import case, check
from whetstone.kit import CharDFA

FN = "schema_to_regex"
DIGITS = "0123456789"
HEX = "0123456789abcdefABCDEF"


# -- the specification: which positions can a value of the schema end at -----------------


def number_end(text, i, fraction):
    """End of the longest JSON number (or integer) starting at i, or None."""
    j = i + 1 if text.startswith("-", i) else i
    if text.startswith("0", j):
        j += 1
    elif j < len(text) and text[j] in "123456789":
        while j < len(text) and text[j] in DIGITS:
            j += 1
    else:
        return None
    if not fraction:
        return j
    if text.startswith(".", j) and j + 1 < len(text) and text[j + 1] in DIGITS:
        j += 1
        while j < len(text) and text[j] in DIGITS:
            j += 1
    if j < len(text) and text[j] in "eE":
        k = j + 1
        if k < len(text) and text[k] in "+-":
            k += 1
        if k < len(text) and text[k] in DIGITS:
            while k < len(text) and text[k] in DIGITS:
                k += 1
            j = k
    return j


def string_end(text, i, low, high):
    """End of the JSON string starting at i if its item count is in [low, high], or None."""
    if not text.startswith('"', i):
        return None
    j, items = i + 1, 0
    while j < len(text) and text[j] != '"':
        if text[j] == "\\":
            if text.startswith("u", j + 1):
                digits = text[j + 2 : j + 6]
                if len(digits) != 4 or any(d not in HEX for d in digits):
                    return None
                j += 6
            elif j + 1 < len(text) and text[j + 1] in '"\\/bfnrt':
                j += 2
            else:
                return None
        elif ord(text[j]) < 0x20:
            return None
        else:
            j += 1
        items += 1
    if j >= len(text) or items < low or (high is not None and items > high):
        return None
    return j + 1


def ends(schema, text, i):
    """The set of positions at which a value of `schema` that starts at `i` can end."""
    if "enum" in schema:
        spellings = [json.dumps(value) for value in schema["enum"]]
        return {i + len(s) for s in spellings if text.startswith(s, i)}
    kind = schema["type"]
    if kind == "null":
        return {i + 4} if text.startswith("null", i) else set()
    if kind == "boolean":
        return {i + len(w) for w in ("true", "false") if text.startswith(w, i)}
    if kind in ("integer", "number"):
        end = number_end(text, i, fraction=kind == "number")
        return set() if end is None else {end}
    if kind == "string":
        end = string_end(text, i, schema.get("minLength", 0), schema.get("maxLength"))
        return set() if end is None else {end}
    if kind == "array":
        if not text.startswith("[", i):
            return set()
        low, high = schema.get("minItems", 0), schema.get("maxItems")
        out = set()
        if low == 0 and text.startswith("]", i + 1):
            out.add(i + 2)
        frontier, count = ends(schema["items"], text, i + 1), 1
        while frontier and (high is None or count <= high):
            if count >= low:
                out |= {j + 1 for j in frontier if text.startswith("]", j)}
            starts = {j + 1 for j in frontier if text.startswith(",", j)}
            frontier = set().union(*(ends(schema["items"], text, j) for j in starts))
            count += 1
        return out
    if kind == "object":
        if not text.startswith("{", i):
            return set()
        required = set(schema.get("required", ()))
        states = {(i + 1, False)}  # (position, whether a member has been written)
        for name, sub in schema.get("properties", {}).items():
            after = set() if name in required else set(states)
            for j, written in states:
                key = ("," if written else "") + json.dumps(name) + ":"
                if text.startswith(key, j):
                    after |= {(e, True) for e in ends(sub, text, j + len(key))}
            states = after
        return {j + 1 for j, _ in states if text.startswith("}", j)}
    raise AssertionError(f"test schema outside the subset: {schema!r}")


def conforms(schema, text):
    return len(text) in ends(schema, text, 0)


# -- generating documents ----------------------------------------------------------------

PLAIN = "abcxyz AZ09_-.,:{}[]"
WILD = ['"', "\\", "/", "\n", "\t", "\r", "\b", "\f", "\x01", "\x1f", "é", "€", "😀", "\x7f"]
NUMBERS = [0, 7, -3, 120, 1.5, -0.25, 1e-05, 2.5e16, 1e22, 100.0, -0.0, 3.14159]


def instance(schema, rng):
    """A random value that satisfies `schema`."""
    if "enum" in schema:
        return schema["enum"][int(rng.integers(len(schema["enum"])))]
    kind = schema["type"]
    if kind == "null":
        return None
    if kind == "boolean":
        return bool(rng.integers(2))
    if kind == "integer":
        return int(rng.choice([0, 1, -1, 7, 10, 42, -305, 2**40, int(rng.integers(-999, 999))]))
    if kind == "number":
        return NUMBERS[int(rng.integers(len(NUMBERS)))]
    if kind == "string":
        low = schema.get("minLength", 0)
        high = schema.get("maxLength", low + 6)
        chars = list(PLAIN) + (WILD if rng.random() < 0.5 else [])
        return "".join(rng.choice(chars, size=int(rng.integers(low, high + 1))))
    if kind == "array":
        low = schema.get("minItems", 0)
        high = schema.get("maxItems", low + 3)
        return [instance(schema["items"], rng) for _ in range(int(rng.integers(low, high + 1)))]
    required = set(schema.get("required", ()))
    return {
        name: instance(sub, rng)
        for name, sub in schema.get("properties", {}).items()
        if name in required or rng.random() < 0.5
    }


def dump(value, rng):
    """Compact JSON, with non-ASCII characters written raw or as \\u escapes."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=bool(rng.integers(2)))


SWAPS = [("true", "True"), ("null", "nul"), (":", ": "), (",", ", "), (",", ",,"), ('"', "'"),
         ("{", "{ "), ("}", ",}"), ("]", ",]"), ("[", "[,"), ("0", "00"), ("-", "+"), (".", ".."),
         ("e", "e+"), ("\\", ""), ("\\n", "\n"), ("1", "01"), ("7", "٧"), (":", "="),
         ('":', '" :'), ("false", "0"), ("[", "[["), ("}", "}}"), ('"', '\\"')]  # fmt: skip


def corrupted(text, rng, n):
    """Near-misses of a document: most of them no longer conform, some still do."""
    out = [text + " ", " " + text, text[:-1], text[1:], text + text, text + ","]
    for _ in range(n):
        at = int(rng.integers(0, len(text)))
        out += [text[:at] + text[at + 1 :], text[:at] + text[at] + text[at:]]
        old, new = SWAPS[int(rng.integers(len(SWAPS)))]
        spots = [m.start() for m in re.finditer(re.escape(old), text)]
        if spots:
            spot = spots[int(rng.integers(len(spots)))]
            out.append(text[:spot] + new + text[spot + len(old) :])
    return out


# -- checking a pattern ------------------------------------------------------------------


def compiled(ctx, schema, *, dfa=True):
    """The solution's pattern for `schema`, compiled; optionally also checked with the kit."""
    pattern = ctx.solution.schema_to_regex(schema)
    check.that(
        isinstance(pattern, str),
        f"{FN} must return a str",
        f"got {type(pattern).__name__}",
    )
    try:
        regex = re.compile(pattern)
    except re.error as error:
        check.that(
            False,
            f"{FN}: the returned pattern is not a valid regular expression for re",
            f"schema: {schema!r}\npattern: {pattern!r}\nre.error: {error}",
        )
    if dfa:
        try:
            CharDFA.from_regex(pattern)
        except ValueError as error:
            check.that(
                False,
                f"{FN}: the returned pattern is outside the regex subset of regex-to-dfa",
                f"schema: {schema!r}\npattern: {pattern!r}\n{error}",
            )
    return regex


def check_document(regex, schema, text, expect=None):
    want = conforms(schema, text)
    if expect is not None and want != expect:
        raise AssertionError(f"fixture verdict is wrong for {text!r} under {schema!r}")
    got = regex.fullmatch(text) is not None
    if got == want:
        return
    detail = f"schema: {schema!r}\ndocument: {text!r}\npattern: {regex.pattern!r}"
    if got:
        message = f"{FN}: the pattern matches a document that does not conform to the schema"
    else:
        message = f"{FN}: the pattern does not match a document that conforms to the schema"
    check.that(False, message, detail)


def check_fixture(ctx, schema, good, bad, *, dfa=True):
    regex = compiled(ctx, schema, dfa=dfa)
    for text in good:
        check_document(regex, schema, text, expect=True)
    for text in bad:
        check_document(regex, schema, text, expect=False)
    return regex


def check_generated(ctx, schema, n=12, *, dfa=True):
    regex = compiled(ctx, schema, dfa=dfa)
    valid = 0
    for _ in range(n):
        text = dump(instance(schema, ctx.rng), ctx.rng)
        # An instance conforms by construction, with one exception: an astral character
        # written as two \u escapes counts as two items and can exceed a maxLength.
        valid += conforms(schema, text)
        check_document(regex, schema, text)
        for near in corrupted(text, ctx.rng, 4):
            check_document(regex, schema, near)
    if valid < n // 2:
        raise AssertionError(f"the generator rarely satisfies {schema!r}")


# -- schemas -----------------------------------------------------------------------------

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    },
    "required": ["name", "age"],
}

TICKET = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "status": {"enum": ["open", "closed", "won't fix"]},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 8}, "maxItems": 3},
        "owner": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "admin": {"type": "boolean"},
            },
            "required": ["name"],
        },
        "score": {"type": "number"},
        "parent": {"enum": [None, 1, 2]},
    },
    "required": ["id", "status", "owner"],
}

ALL_OPTIONAL = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "null"}},
}


@case("the worked example: an object with two required properties and one optional",
      public=True)  # fmt: skip
def worked_example(ctx):
    check_fixture(
        ctx,
        PERSON,
        good=[
            '{"name":"Ada","age":36}',
            '{"name":"Ada","age":36,"email":"ada@example.com"}',
            '{"name":"","age":-1}',
        ],
        bad=[
            '{"name": "Ada", "age": 36}',
            '{"age":36,"name":"Ada"}',
            '{"name":"Ada"}',
            '{"name":"Ada","age":36,}',
            '{"name":"Ada","age":"36"}',
            '{"name":"Ada","age":36.0}',
            '{"name":"Ada","age":36,"email":"a","x":1}',
            '{"name":"Ada","age":036}',
            "{}",
            '{"name":"Ada","email":"a","age":36}',
        ],
    )


# Rejects leading_zeros, exponent_needs_sign, unicode_digits, and enum_not_json.
@case("scalars: integer, number, boolean, null, and enum", public=True)
def scalars(ctx):
    check_fixture(ctx, {"type": "integer"}, ["0", "-0", "7", "-12", "1000000"],
                  ["", "-", "01", "+1", "1.0", "1e3", " 1", "1 ", "--1", "٣", "1٣", "4２", "0x10",
                   "1_000"])  # fmt: skip
    check_fixture(ctx, {"type": "number"},
                  ["0", "-0", "3.25", "-0.5", "1e5", "1E5", "1e+5", "2.5e-10", "10", "0.0", "1e05"],
                  ["", ".5", "1.", "01.5", "1e", "1e+", "1.e5", "+1", "NaN", "Infinity", "1,5",
                   "0x1p3", "1.5.2", "1e5.5", "٣.٥", "1.٥", "1e٥", "- 1"])  # fmt: skip
    check_fixture(ctx, {"type": "boolean"}, ["true", "false"],
                  ["True", "FALSE", "1", '"true"', "", "truefalse", "tru"])  # fmt: skip
    check_fixture(ctx, {"type": "null"}, ["null"], ["", "None", "nul", "NULL", '"null"', "nulll"])
    check_fixture(ctx, {"enum": ["red", "green", 3, True, None]},
                  ['"red"', '"green"', "3", "true", "null"],
                  ["red", '"blue"', "4", "false", '"3"', '"red"|"green"', "", "True", "None",
                   '"redgreen"', '"red""green"'])  # fmt: skip


# Rejects strings_unescaped and control_chars_allowed.
@case("strings: escapes, control characters, and length bounds", public=True)
def strings(ctx):
    check_fixture(
        ctx,
        {"type": "string"},
        good=[
            '""',
            '"abc"',
            '"a b, c: {d} [e]"',
            r'"say \"hi\""',
            r'"back\\slash"',
            r'"a\/b"',
            r'"tab\tnewline\n\r\b\f"',
            r'"\u00e9\u20AC"',
            '"é€😀"',
            r'"\\"',
            '"\x7f"',
            r'"\ud83d\ude00"',
            '"\'"',
        ],
        bad=[
            '"',
            '"abc',
            'abc"',
            "abc",
            "'abc'",
            '"a"b"',
            r'"\"',
            r'"a\"',
            '"a\nb"',
            '"a\tb"',
            '"\x00"',
            '"\x1f"',
            r'"\x41"',
            r'"\a"',
            r'"\u12"',
            r'"\u12G4"',
            r'"\U0041"',
            '"a""b"',
            '"a" ',
            r'"\\\"',
        ],
    )
    check_fixture(
        ctx,
        {"type": "string", "minLength": 2, "maxLength": 4},
        good=['"ab"', '"abcd"', r'"\n\t"', r'"a\u00e9"', '"é€😀x"', r'"\\\\"', r'"\"\"\"\""'],
        bad=['""', '"a"', '"abcde"', r'"\n"', r'"\u00e9"', r'"\n\t\r\b\f"', '"é"'],
    )
    check_fixture(ctx, {"type": "string", "minLength": 3}, ['"abc"', '"abcdefghij"'],
                  ['"ab"', '""'])  # fmt: skip
    check_fixture(ctx, {"type": "string", "maxLength": 0}, ['""'], ['"a"', r'"\n"'])


# Rejects max_items_off_by_one and empty_array_rejected.
@case("arrays: items, minItems, maxItems, and the empty array", public=True)
def arrays(ctx):
    ints = {"type": "integer"}
    check_fixture(ctx, {"type": "array", "items": ints}, ["[]", "[1]", "[1,2,3]", "[0,-1,22]"],
                  ["[", "[1,]", "[,1]", "[1 ,2]", "[1, 2]", "[1,,2]", "[1,2.5]", '["1"]', "[[1]]",
                   "[] ", "1"])  # fmt: skip
    check_fixture(ctx, {"type": "array", "items": ints, "minItems": 2, "maxItems": 4},
                  ["[1,2]", "[1,2,3]", "[1,2,3,4]"], ["[]", "[1]", "[1,2,3,4,5]", "[1,2,]"])  # fmt: skip
    check_fixture(ctx, {"type": "array", "items": ints, "maxItems": 2}, ["[]", "[1]", "[1,2]"],
                  ["[1,2,3]", "[,]"])  # fmt: skip
    check_fixture(ctx, {"type": "array", "items": ints, "minItems": 1}, ["[1]", "[1,2,3,4,5,6]"],
                  ["[]"])  # fmt: skip
    check_fixture(ctx, {"type": "array", "items": ints, "maxItems": 0}, ["[]"], ["[1]", "[ ]"])
    check_fixture(ctx, {"type": "array", "items": ints, "minItems": 1, "maxItems": 1}, ["[5]"],
                  ["[]", "[5,6]"])  # fmt: skip
    nested = {"type": "array", "items": {"type": "array", "items": {"type": "boolean"},
                                         "maxItems": 2}, "minItems": 1}  # fmt: skip
    check_fixture(ctx, nested, ["[[]]", "[[true],[false,true],[]]"],
                  ["[]", "[[true,true,true]]", "[true]", "[[],]"])  # fmt: skip


# Rejects dangling_comma and required_ignored: every subset of optional properties, with
# the commas where they belong.
@case("objects: optional properties may be left out, and no comma is left behind",
      public=True)  # fmt: skip
def optional_properties(ctx):
    check_fixture(
        ctx,
        ALL_OPTIONAL,
        good=[
            "{}",
            '{"a":1}',
            '{"b":true}',
            '{"c":null}',
            '{"a":1,"b":true}',
            '{"a":1,"c":null}',
            '{"b":false,"c":null}',
            '{"a":1,"b":true,"c":null}',
        ],
        bad=[
            '{"a":1,}',
            '{,"b":true}',
            '{"a":1,,"c":null}',
            '{"b":true,"a":1}',
            '{"a":1"b":true}',
            '{"c":null,"c":null}',
            "{,}",
            '{"a":1,"b":true,}',
            "{ }",
            '{"a":1,"a":2}',
        ],
    )
    last_optional = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "z": {"type": "integer"},
        },
        "required": ["y"],
    }
    check_fixture(
        ctx,
        last_optional,
        good=['{"y":2}', '{"x":1,"y":2}', '{"y":2,"z":3}', '{"x":1,"y":2,"z":3}'],
        bad=[
            "{}",
            '{"x":1}',
            '{"x":1,"z":3}',
            '{"y":2,}',
            '{,"y":2}',
            '{"x":1,"y":2,}',
            '{"x":1,,"y":2}',
            '{"y":2"z":3}',
            '{"z":3,"y":2}',
        ],
    )
    check_fixture(ctx, {"type": "object"}, ["{}"], ['{"a":1}', "{ }", "", "{}{}"])
    check_fixture(ctx, {"type": "object", "properties": {}}, ["{}"], ['{"a":1}'])


# Rejects names_not_escaped: a property name and enum values full of metacharacters.
@case("property names and enum values are matched literally, whatever characters they hold")
def literal_names(ctx):
    schema = {
        "type": "object",
        "properties": {
            "a.b": {"type": "integer"},
            "price ($)": {"type": "number"},
            "c|d": {"enum": ["x.y", "a+b", "(z)", "[k]", "q?", "back\\slash", 'quo"te', "^$", "*"]},
            "e{2}": {"type": "null"},
        },
        "required": ["a.b"],
    }
    check_fixture(
        ctx,
        schema,
        good=[
            '{"a.b":1}',
            '{"a.b":1,"price ($)":2.5}',
            '{"a.b":1,"c|d":"x.y"}',
            '{"a.b":1,"c|d":"a+b","e{2}":null}',
            '{"a.b":1,"c|d":"(z)"}',
            '{"a.b":1,"c|d":"q?"}',
            r'{"a.b":1,"c|d":"back\\slash"}',
            r'{"a.b":1,"c|d":"quo\"te"}',
            '{"a.b":1,"c|d":"^$"}',
            '{"a.b":1,"c|d":"*"}',
            '{"a.b":1,"c|d":"[k]"}',
        ],
        bad=[
            '{"axb":1}',
            '{"a.b":1,"price ":2.5}',
            '{"a.b":1,"c":"x.y"}',
            '{"a.b":1,"d":"x.y"}',
            '{"a.b":1,"c|d":"xzy"}',
            '{"a.b":1,"c|d":"aab"}',
            '{"a.b":1,"c|d":"z"}',
            '{"a.b":1,"c|d":"k"}',
            '{"a.b":1,"c|d":"q"}',
            '{"a.b":1,"ee":null}',
            '{"a.b":1,"c|d":""}',
            '{"a.b":1,"c|d":"a++b"}',
        ],
    )
    check_generated(ctx, schema)


@case("a nested schema: generated instances match, corrupted copies get the right verdict")
def nested_generated(ctx):
    check_generated(ctx, TICKET, n=20)
    check_fixture(
        ctx,
        TICKET,
        good=[
            '{"id":1,"status":"open","owner":{"name":"x"}}',
            '{"id":1,"status":"won\'t fix","tags":[],"owner":{"name":"x","admin":true},"parent":null}',
        ],
        bad=[
            '{"id":1,"status":"open","owner":{"name":""}}',
            '{"id":1,"status":"open","owner":{}}',
            '{"id":1,"status":"open","tags":["a","b","c","d"],"owner":{"name":"x"}}',
            '{"id":1,"status":"open","tags":["toolongtag"],"owner":{"name":"x"}}',
            '{"id":1,"status":"reopened","owner":{"name":"x"}}',
            '{"id":1,"status":"open","owner":{"name":"x"},"parent":3}',
            '{"id":1,"status":"open","owner":{"admin":true,"name":"x"}}',
        ],
    )


@case("every combination of required and optional over four properties")
def required_patterns(ctx):
    names = ["p", "q", "r", "s"]
    kinds = [{"type": "integer"}, {"type": "boolean"}, {"type": "string", "maxLength": 3},
             {"type": "null"}]  # fmt: skip
    for mask in range(16):
        schema = {
            "type": "object",
            "properties": dict(zip(names, kinds, strict=True)),
            "required": [n for bit, n in enumerate(names) if mask >> bit & 1],
        }
        regex = compiled(ctx, schema)
        values = {"p": 5, "q": True, "r": "ab", "s": None}
        for present in range(16):
            chosen = {n: values[n] for bit, n in enumerate(names) if present >> bit & 1}
            text = json.dumps(chosen, separators=(",", ":"))
            check_document(regex, schema, text)
            for near in (text.replace(",", ",,", 1), text[:-1] + ",}", "{," + text[1:]):
                check_document(regex, schema, near)


@case("random schemas of the subset, nested up to three levels")
def random_schemas(ctx):
    def schema_of(rng, depth):
        kinds = ["integer", "number", "boolean", "null", "string", "enum"]
        kind = str(rng.choice(kinds + ["array", "object", "object"] * (depth < 3)))
        if kind == "enum":
            pool = ["a", "b c", "d.e", 0, 12, True, False, None, "x|y"]
            picks = rng.choice(len(pool), size=int(rng.integers(1, 4)), replace=False)
            return {"enum": [pool[int(i)] for i in picks]}
        schema = {"type": kind}
        if kind == "string" and rng.random() < 0.5:
            low = int(rng.integers(0, 3))
            schema.update(minLength=low, maxLength=low + int(rng.integers(0, 4)))
        if kind == "array":
            schema["items"] = schema_of(rng, depth + 1)
            if rng.random() < 0.6:
                schema["minItems"] = int(rng.integers(0, 3))
            if rng.random() < 0.6:
                schema["maxItems"] = schema.get("minItems", 0) + int(rng.integers(0, 3))
        if kind == "object":
            names = [str(n) for n in rng.choice(["id", "name", "k_1", "a b", "v"],
                                                size=int(rng.integers(0, 5)), replace=False)]  # fmt: skip
            schema["properties"] = {name: schema_of(rng, depth + 1) for name in names}
            schema["required"] = [name for name in names if rng.random() < 0.5]
        return schema

    def prop(rng):
        check_generated(ctx, schema_of(rng, 0), n=6, dfa=False)

    check.forall(prop, trials=25)


@case("large bounds: maxLength 100, maxItems 100, and twelve optional properties")
def large_bounds(ctx):
    schema = {"type": "string", "minLength": 50, "maxLength": 100}
    check_fixture(ctx, schema, ['"' + "ab" * 25 + '"', '"' + "x" * 100 + '"'],
                  ['"' + "x" * 49 + '"', '"' + "x" * 101 + '"'], dfa=False)  # fmt: skip
    schema = {"type": "array", "items": {"type": "boolean"}, "minItems": 99, "maxItems": 100}
    check_fixture(ctx, schema, ["[" + ",".join(["true"] * n) + "]" for n in (99, 100)],
                  ["[" + ",".join(["true"] * n) + "]" for n in (98, 101)], dfa=False)  # fmt: skip
    wide = {"type": "object", "properties": {f"k{i}": {"type": "integer"} for i in range(12)}}
    regex = compiled(ctx, wide, dfa=False)
    for chosen in ([], [0], [11], [0, 11], [3, 4, 9], list(range(12))):
        text = json.dumps({f"k{i}": i for i in chosen}, separators=(",", ":"))
        check_document(regex, wide, text, expect=True)
        check_document(regex, wide, text[:-1] + ",}", expect=False)


@case("a schema outside the subset raises ValueError")
def unsupported(ctx):
    for schema in ({"type": "tuple"}, {}, {"type": ["string", "null"]}, {"type": "any"},
                   {"type": "array", "items": {"type": "date"}},
                   {"type": "object", "properties": {"a": {"minLength": 1}}}):  # fmt: skip
        try:
            ctx.solution.schema_to_regex(schema)
        except ValueError:
            continue
        check.that(
            False,
            f"{FN}: returned a pattern for a schema with an unsupported type",
            f"schema: {schema!r}",
        )
