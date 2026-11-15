"""Tests for structural-tag-dispatch.

The expected answers are computed here by `read`, which reads a whole byte string
from the beginning, position by position, as the statement's four rules say, and
reports whether it is valid so far and whether it is complete. It shares no code and
no state representation with the reference. A token is allowed in a state exactly
when the text so far plus the token's bytes is valid so far.

The dispatcher is built and called directly on the solution module: its states are
opaque objects that must not be copied or converted.
"""

from whetstone.judge import case, check
from whetstone.kit import DEAD, CharDFA, ToyVocab


class Spec:
    """A configuration: trigger strings and (begin, inner regex, end) tags."""

    def __init__(self, name, triggers, tags):
        self.name = name
        self.trigger_strings = list(triggers)
        self.tag_strings = list(tags)
        self.triggers = [t.encode() for t in triggers]
        self.tags = [(b.encode(), CharDFA.from_regex(r), e.encode()) for b, r, e in tags]

    def build(self, ctx, vocab):
        tags = [(b, CharDFA.from_regex(r), e) for b, r, e in self.tag_strings]
        return ctx.solution.TagDispatch(list(self.trigger_strings), tags, vocab)

    def __repr__(self):
        return f"triggers {self.trigger_strings!r}, tags (begin, regex, end) {self.tag_strings!r}"


def read(spec, data):
    """(valid so far, complete) for the bytes `data`, read from the beginning."""
    free_from = 0
    while True:
        # Rule 1: the first position where the current free text ends with a trigger.
        fired = None
        for stop in range(free_from + 1, len(data) + 1):
            for trigger in spec.triggers:
                at = stop - len(trigger)
                if at >= free_from and data[at:stop] == trigger:
                    fired = (at, trigger)
            if fired:
                break
        if fired is None:
            return True, True
        at, trigger = fired
        # Rule 2: the trigger and what follows spell the begin of one of its tags.
        rest = data[at:]
        tags = [tag for tag in spec.tags if tag[0].startswith(trigger)]
        begun = [tag for tag in tags if rest.startswith(tag[0])]
        if not begun:
            return any(tag[0].startswith(rest) for tag in tags), False
        begin, inner, end = begun[0]
        # Rule 3: content, for as long as the automaton can read the next byte.
        i, q = at + len(begin), inner.start
        while i < len(data) and inner._next(q, data[i]) != DEAD:
            q = inner._next(q, data[i])
            i += 1
        if i == len(data):
            return True, False
        # Rule 4: the end string, only from an accepting state.
        tail = data[i : i + len(end)]
        if not inner.is_accepting(q) or tail != end[: len(tail)]:
            return False, False
        if len(tail) < len(end):
            return True, False
        free_from = i + len(end)


def show(vocab, tokens):
    head = ", ".join(f"{t}={vocab.pieces[t]!r}" for t in tokens[:14])
    return (head + (", ..." if len(tokens) > 14 else "")) or "none"


def check_state(dispatch, spec, vocab, state, text):
    """allowed and is_accepting in one state, against `read`. Returns the allowed list."""
    where = f"configuration: {spec!r}\nvocabulary: {vocab!r}\ntext so far: {text!r}\n"
    got = dispatch.allowed(state)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        "allowed: must return a list of Python ints",
        where + f"got {got!r}",
    )
    check.that(
        all(0 <= token < vocab.size for token in got),
        "allowed: the list contains a number that is not a token id",
        where + f"got {got!r}",
    )
    expected = [
        token for token, piece in enumerate(vocab.pieces) if piece and read(spec, text + piece)[0]
    ]
    wrong = sorted(set(got) - set(expected))
    missing = sorted(set(expected) - set(got))
    if any(not vocab.pieces[t] for t in wrong):
        message = "allowed: a special token (a token without bytes) is in the list"
    elif wrong:
        message = "allowed: a token is in the list whose bytes cannot all be read here"
    else:
        message = "allowed: a token whose bytes can all be read here is missing"
    check.that(
        not wrong and not missing,
        message,
        where + f"in your list but not allowed: {show(vocab, wrong)}\n"
        f"allowed but missing from your list: {show(vocab, missing)}\n",
    )
    check.that(
        got == expected, "allowed: the ids are not in increasing order", where + f"got {got!r}"
    )
    accepting = dispatch.is_accepting(state)
    complete = read(spec, text)[1]
    check.that(
        accepting is True or accepting is False,
        "is_accepting: must return True or False",
        where + f"got {accepting!r}",
    )
    if accepting != complete:
        check.that(
            False,
            "is_accepting: true inside an open tag"
            if accepting
            else "is_accepting: false although the text is outside every tag",
            where + f"got {accepting}, expected {complete}\n",
        )
    return got


def follow(ctx, spec, vocab, tokens):
    """Advance through the token ids, checking every state on the way."""
    dispatch = spec.build(ctx, vocab)
    state, text = dispatch.start(), b""
    check_state(dispatch, spec, vocab, state, text)
    for token in tokens:
        piece = vocab.pieces[token]
        if not read(spec, text + piece)[0]:
            raise RuntimeError(f"authoring error: {piece!r} cannot follow {text!r}")
        state = dispatch.advance(state, token)
        text += piece
        check_state(dispatch, spec, vocab, state, text)
    return dispatch, state, text


def singles(vocab, text):
    """One token per byte: every trigger, begin, and end string is split everywhere."""
    return [vocab.id_of(bytes([byte])) for byte in text.encode()]


def follow_text(ctx, spec, vocab, text):
    """Both tokenizations of the same text: byte by byte, and with merged pieces."""
    follow(ctx, spec, vocab, singles(vocab, text))
    merged = vocab.encode(text)
    if merged != singles(vocab, text):
        follow(ctx, spec, vocab, merged)


JSON_CALLS = Spec(
    "JSON-style calls",
    ['{"fn"'],
    [('{"fn": "neg", "x": ', r"-?\d+", "}"), ('{"fn": "say", "text": ', r'"[a-z ]*"', "}")],
)
FUNCTIONS = Spec(
    "function tags",
    ["<fn="],
    [("<fn=add>", r"\d+,\d+", "</fn>"), ("<fn=neg>", r"-?\d+", "</fn>"),
     ("<fn=now>", r"", "</fn>"), ("<fn=nest>", r"\[(true|false|null)(, (true|false|null))*\]",
                                  "</fn>")],
)  # fmt: skip
TWO_TRIGGERS = Spec(
    "two triggers, one of them with two tags",
    ["<tool_call>", "<think"],
    [("<tool_call>", r'\{"name": "[a-z_]+", "arguments": \{("[a-z]+": -?\d+)?\}\}',
      "</tool_call>"),
     ("<think>", r"[a-z ,.]*", "</think>"), ("<think fast>", r"[a-z]*", "</think>")],
)  # fmt: skip
OVERLAP = Spec("a trigger that overlaps itself", ["<<fn=", "aab"],
               [("<<fn=id>", r"\d", ";"), ("aab:", r"[a-c]+", ".")])  # fmt: skip
FORBIDDEN = Spec("a trigger without tags", ["<fn=", "<|", "DROP"],
                 [("<fn=id>", r"\d+", "</fn>")])  # fmt: skip
FENCE = Spec("fenced blocks", ["```json\n"],
             [("```json\n", r"\[\d(, \d)*\]", "\n```")])  # fmt: skip
SMALL = Spec("the 64-token vocabulary", ['{"', "[["],
             [('{"a": ', r"(true|false|null)", "}"), ('{"a", ', r"\d+", ' "}'),
              ("[[", r"\d(, \d)*", "]]")])  # fmt: skip


@case('the worked example: JSON-style calls, \'Hi {"fn": "neg", "x": -4}\'', public=True)
def worked_example(ctx):
    vocab = ToyVocab(size=132, seed=0)
    dispatch = JSON_CALLS.build(ctx, vocab)
    everything = [t for t in range(vocab.size) if vocab.pieces[t]]
    digits = [vocab.id_of(str(d).encode()) for d in range(10)]

    def ids(*pieces):
        return sorted(vocab.id_of(p) for p in pieces)

    steps = [
        ("", everything, True),
        ('Hi {"fn', [t for t in everything if vocab.pieces[t] != b'"}'], True),
        ('"', ids(b":"), False),
        (': "', ids(b"n", b"s"), False),
        ("n", ids(b"e"), False),
        ('eg", "x": ', sorted([*digits, vocab.id_of(b"-")]), False),
        ("-4", sorted([*digits, vocab.id_of(b"}")]), False),
        ("}", everything, True),
    ]
    state, text = dispatch.start(), ""
    for more, expected, accepting in steps:
        for token in vocab.encode(more):
            state = dispatch.advance(state, token)
        text += more
        check.equal(dispatch.allowed(state), expected, label=f"allowed after {text!r}")
        check.equal(dispatch.is_accepting(state), accepting, label=f"is_accepting after {text!r}")


@case("a trigger fires whether it arrives byte by byte or inside merged tokens", public=True)
def trigger_across_tokens(ctx):
    # Rejects single_token_trigger: with one token per byte no token contains a trigger.
    vocab = ToyVocab(size=132, seed=1)
    follow_text(ctx, FUNCTIONS, vocab, "ok <fn=neg>-12</fn> done")
    follow_text(ctx, JSON_CALLS, vocab, 'so {"fn": "say", "text": "true false"} ok')
    follow_text(ctx, TWO_TRIGGERS, vocab, 'a <tool_call>{"name": "f", "arguments": {}}</tool_call>')


@case("several tags behind one trigger: candidates narrow byte by byte", public=True)
def tags_share_a_trigger(ctx):
    # Rejects first_tag_only and begin_not_narrowed: after '<fn=n' only 'neg', 'now',
    # and 'nest' remain, and after '<fn=ne' the next byte is 'g' or 's', not 'o'.
    vocab = ToyVocab(size=132, seed=0)
    for text in (
        "<fn=add>1,2</fn>",
        "<fn=now></fn>",
        "<fn=nest>[true, null]</fn>",
        "<fn=neg>7</fn>",
    ):
        follow_text(ctx, FUNCTIONS, vocab, "x " + text)
    follow_text(ctx, TWO_TRIGGERS, vocab, "<think fast>hm</think><think>so, yes.</think>")


@case("the output is complete only outside tags; a tag ends with its end string", public=True)
def tags_must_close(ctx):
    # Rejects accepting_inside_tag and end_optional. Free text that stops in the
    # middle of a possible trigger is complete; everything after a trigger is not,
    # until the last byte of the end string.
    vocab = ToyVocab(size=132, seed=2)
    follow_text(ctx, FUNCTIONS, vocab, "<fn=add>10,20</fn>")
    follow_text(ctx, FUNCTIONS, vocab, "see <fn or <f= no <fn")
    follow_text(ctx, JSON_CALLS, vocab, 'a {"fn": "neg", "x": 0} b {"f')


@case("a second call of the same tag starts its content from the beginning", public=True)
def second_call(ctx):
    # Rejects inner_not_reset.
    vocab = ToyVocab(size=132, seed=0)
    follow_text(
        ctx, FUNCTIONS, vocab, "<fn=neg>-1</fn> and <fn=neg>-2</fn><fn=add>3,4</fn><fn=add>5"
    )
    follow_text(ctx, JSON_CALLS, vocab, '{"fn": "say", "text": "a"}{"fn": "say", "text": "b"}')


@case("a trigger that overlaps its own beginning: '<<fn=' inside '<<<fn='")
def self_overlap(ctx):
    # Rejects no_restart.
    vocab = ToyVocab(size=132, seed=3)
    for text in ("<<<fn=id>5; <<f<<<<fn=id>6;", "aaab:abc. ab aaaab:c.", "<<aab:a.<<fn=id>1;"):
        follow_text(ctx, OVERLAP, vocab, text)


@case("ambiguous configurations raise ValueError; unusual valid ones do not")
def construction(ctx):
    # Rejects no_validation.
    vocab = ToyVocab(size=132, seed=0)
    number = r"\d+"
    bad = [
        (["<fn=", "<fn=add"], [("<fn=add>", number, ";")]),  # a trigger is a prefix of another
        (["<fn=", "n="], [("<fn=a>", number, ";")]),  # ... a suffix of another
        (["<call>", "al"], [("<call>", number, ";")]),  # ... in the middle of another
        (["<a>", "<a>"], [("<a>", number, ";")]),  # the same trigger twice
        (["<a>"], [("<a>", number, ";"), ("<b>", number, ";")]),  # a begin without a trigger
        (["<a"], [("<a>", number, ";"), ("<a>x", number, ";")]),  # a begin prefixes another
        (["<a"], [("<a>", number, ";"), ("<a>", r"[a-z]", ".")]),  # the same begin twice
        ([], [("<a>", number, ";")]),
    ]
    for triggers, tags in bad:
        built = [(b, CharDFA.from_regex(r), e) for b, r, e in tags]
        check.raises(
            ValueError, ctx.solution.TagDispatch, triggers, built, vocab,
            label="TagDispatch with an ambiguous configuration",
        )  # fmt: skip
    # Fine: no tags at all; a begin equal to its trigger; triggers that share a prefix
    # or overlap end to start without one containing the other.
    for spec in (
        Spec("no triggers", [], []),
        Spec("prefix-sharing triggers", ["<tool_a", "<tool_b", "b>x"],
             [("<tool_a", r"\d", ">"), ("<tool_b>", r"\d", ";")]),
    ):  # fmt: skip
        follow_text(ctx, spec, vocab, "<tool_ <tool_b>4; <tool_a1> b> <tool")


@case("a trigger to which no tag belongs can never be completed")
def trigger_without_tags(ctx):
    # Rejects dead_trigger_ignored: after '<' the token '|' is not allowed.
    vocab = ToyVocab(size=132, seed=4)
    follow_text(ctx, FORBIDDEN, vocab, "a < b, DRO, <fn=id>1</fn> DR<")


@case("after a tag ends the trigger scan starts afresh")
def scan_starts_afresh(ctx):
    # Rejects scan_spans_tags: the end string finishes with three backticks, which is
    # how the trigger begins, but 'json' after a closed block is just text.
    vocab = ToyVocab(size=132, seed=5)
    follow_text(ctx, FENCE, vocab, "```json\n[1, 2]\n```json\n and ``` then ```json\n[3]\n```")


@case("special tokens are never allowed; advance rejects what allowed does not list")
def advance_contract(ctx):
    # Rejects specials_allowed.
    vocab = ToyVocab(size=132, seed=0)
    dispatch = FUNCTIONS.build(ctx, vocab)
    start = dispatch.start()
    fired = start
    for token in singles(vocab, "<fn="):
        fired = dispatch.advance(fired, token)
    for state, text in ((start, b""), (fired, b"<fn=")):
        got = check_state(dispatch, FUNCTIONS, vocab, state, text)
        for token in range(vocab.size):
            if token not in got:
                check.raises(
                    ValueError, dispatch.advance, state, token,
                    label="advance with a token that is not allowed",
                )  # fmt: skip
    try:
        hash(fired)
    except TypeError:
        check.that(False, "a state is not hashable", f"state: {fired!r}")
    # Branch twice from one state; the state itself must be unchanged afterwards.
    one = dispatch.advance(fired, vocab.id_of(b"a"))
    other = dispatch.advance(fired, vocab.id_of(b"n"))
    check_state(dispatch, FUNCTIONS, vocab, one, b"<fn=a")
    check_state(dispatch, FUNCTIONS, vocab, other, b"<fn=n")
    check_state(dispatch, FUNCTIONS, vocab, fired, b"<fn=")


@case("the 64-token vocabulary, where merged pieces cross every boundary, and all 282 tokens")
def other_vocabularies(ctx):
    for seed in (0, 7):
        vocab = ToyVocab(size=64, seed=seed)
        for text in ('{"a": true} {"a", 12 "}', "[[1, 2]] [1] [[3]]", '{ "e": 1} {"a": null}{"'):
            follow(ctx, SMALL, vocab, vocab.encode(text))
    vocab = ToyVocab(size=282, seed=8)
    follow(ctx, JSON_CALLS, vocab, vocab.encode('é {"fn": "neg", "x": -7} €'))


@case("property: random documents under random tokenizations")
def random_documents(ctx):
    specs = [JSON_CALLS, FUNCTIONS, TWO_TRIGGERS, OVERLAP, FORBIDDEN, FENCE]
    free = ["", " ", "ok ", "a < b ", "<", "<f", "<fn", "{", '{"f', '{"fn', "<<", "aa", "``` ",
            "true, false ", "<tool_", "<thin", "DR", "x=1 ", "\n"]  # fmt: skip
    contents = {
        r"-?\d+": ["0", "-12", "407"], r'"[a-z ]*"': ['""', '"hi there"', '"null"'],
        r"\d+,\d+": ["1,2", "30,4"], r"": [""], r"\d": ["7"], r"[a-c]+": ["abc", "c"],
        r"\[(true|false|null)(, (true|false|null))*\]": ["[true]", "[null, false]"],
        r'\{"name": "[a-z_]+", "arguments": \{("[a-z]+": -?\d+)?\}\}':
            ['{"name": "f", "arguments": {}}', '{"name": "get_x", "arguments": {"n": -3}}'],
        r"[a-z ,.]*": ["", "well, yes."], r"[a-z]*": ["", "go"], r"\d+": ["1", "22"],
        r"\[\d(, \d)*\]": ["[1]", "[1, 2, 3]"],
    }  # fmt: skip

    def prop(rng):
        spec = specs[int(rng.integers(len(specs)))]
        vocab = ToyVocab(size=132, seed=int(rng.integers(40)))
        # A valid document: free text and tags in turn, possibly cut off at the end.
        data = b""
        for _ in range(int(rng.integers(1, 4))):
            piece = free[int(rng.integers(len(free)))].encode()
            if read(spec, data + piece) == (True, True):
                data += piece
            if spec.tag_strings and rng.random() < 0.8:
                begin, regex, end = spec.tag_strings[int(rng.integers(len(spec.tag_strings)))]
                options = contents[regex]
                call = (begin + options[int(rng.integers(len(options)))] + end).encode()
                if read(spec, data + call) == (True, True):
                    data += call
        if rng.random() < 0.4:
            data = data[: int(rng.integers(len(data) + 1))]
        if not read(spec, data)[0]:
            raise RuntimeError(f"authoring error: generated an invalid document {data!r}")
        # A random tokenization of it: any token that spells the next bytes.
        tokens, at = [], 0
        while at < len(data):
            fits = [t for t, p in enumerate(vocab.pieces) if p and data.startswith(p, at)]
            token = fits[int(rng.integers(len(fits)))]
            tokens.append(token)
            at += len(vocab.pieces[token])
        dispatch, state, text = follow(ctx, spec, vocab, tokens)
        # Then leave the script: a few allowed tokens at random.
        for _ in range(4):
            allowed = check_state(dispatch, spec, vocab, state, text)
            if not allowed:
                break
            token = int(allowed[int(rng.integers(len(allowed)))])
            state = dispatch.advance(state, token)
            text += vocab.pieces[token]

    check.forall(prop, trials=30)
