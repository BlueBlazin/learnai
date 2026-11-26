"""Tests for grammar-token-mask.

The expected answers come from the kit's oracle, an Earley recognizer that is run
from scratch on `text so far + token bytes` for every token: a token is allowed
exactly when `grammar.accepts_prefix` says so, and a state is accepting exactly when
`grammar.accepts(text so far)`. That is slow by design (it is what the matcher is
for), so the vocabularies are small and the texts short. Most cases check every token
in every state they visit; the random walks check every token the solution allows
and a sample of those it does not.

The matcher is built and called directly on the solution module: its states are
opaque objects that must not be copied or converted.
"""

from learnai.judge import case, check
from learnai.kit import Grammar, ToyVocab

LISTS = """
    root  ::= "[" ws (value ("," ws value)*)? "]"
    value ::= [0-9]+ | "true" | "false" | root      # nested lists
    ws    ::= " "?
"""

# Left recursion, direct (expr, term) and through a rule that can be empty (list).
EXPR = """
    root ::= expr
    expr ::= expr "-" term | term
    term ::= term "." atom | atom
    atom ::= [0-9]+ | "[" expr "]"
"""
HIDDEN_LEFT = """
    root ::= list
    list ::= ws list "," | ws [0-9]
    ws   ::= " "*
"""

JSON = r"""
    root   ::= ws value ws
    value  ::= object | array | string | number | "true" | "false" | "null"
    object ::= "{" ws (member ("," ws member)*)? ws "}"
    member ::= string ws ":" ws value ws
    array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
    string ::= "\"" ([^"\\\x00-\x1F] | "\\" ["\\nt])* "\""
    number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)?
    ws     ::= " "*
"""

# Alternatives with common beginnings, and two derivations of the same text.
ALIKE = """
    root ::= word ("," " "? word)*
    word ::= "tr" | "true" | "tru" "e"? | "t" tail | [a-f]+ "l"?
    tail ::= "r" | "rue" | "ru" "st"
"""

UNICODE = r"""
    root ::= (sym | "<" [^<>\n]* ">" | "{" [^\x00-\x7F]+ "}")+
    sym  ::= "é" | "€" | "😀" | [à-ÿ] | [0-9]
"""

# The same empty rule is needed several times at one position.
NULLABLE = """
    root  ::= "[" ws items ws "]" tail
    items ::= (item ws)*
    item  ::= [0-9] | "{" ws "}"
    ws    ::= " "?
    tail  ::= ws ws ("." ws)?
"""

COUNTS = """
    root ::= [0-9]{2,4} ("-" [0-9]{2}){0,2} "e"?
"""

STRING = r"""
    root ::= "\"" [^"\\]* "\""
"""

GRAMMARS = {
    "nested lists": LISTS,
    "left-recursive arithmetic": EXPR,
    "left recursion behind an empty rule": HIDDEN_LEFT,
    "JSON": JSON,
    "alternatives that begin alike": ALIKE,
    "non-ASCII characters": UNICODE,
    "empty rules": NULLABLE,
    "bounded repetition": COUNTS,
}


def show(vocab, tokens):
    head = ", ".join(f"{t}={vocab.pieces[t]!r}" for t in tokens[:16])
    return (head + (", ..." if len(tokens) > 16 else "")) or "none"


def describe(name, vocab, text):
    return f"grammar: {name}\nvocabulary: {vocab!r}\ntext so far: {text!r}\n"


def check_state(matcher, grammar, name, vocab, state, text, *, rng=None, sample=10):
    """allowed and is_accepting in one state, against the oracle.

    With `rng`, every allowed token and `sample` of the others are checked instead of
    the whole vocabulary. Returns the solution's allowed list.
    """
    got = matcher.allowed(state)
    where = describe(name, vocab, text)
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
    if rng is None:
        candidates = range(vocab.size)
    else:
        others = [t for t in range(vocab.size) if t not in set(got)]
        picked = rng.choice(len(others), size=min(sample, len(others)), replace=False)
        candidates = sorted({*got, *(others[int(i)] for i in picked)})
    viable = {
        token
        for token in candidates
        if vocab.pieces[token] and grammar.accepts_prefix(text + vocab.pieces[token])
    }
    wrong = sorted(set(got) - viable)
    missing = sorted(viable - set(got))
    specials = [t for t in wrong if not vocab.pieces[t]]
    if specials:
        message = "allowed: a special token (a token without bytes) is in the list"
    elif wrong:
        message = "allowed: a token is in the list that takes the text outside the language"
    else:
        message = "allowed: a token that keeps the text inside the language is missing"
    check.that(
        not wrong and not missing,
        message,
        where + f"in your list but not allowed: {show(vocab, wrong)}\n"
        f"allowed but missing from your list: {show(vocab, missing)}\n",
    )
    check.that(
        got == sorted(set(got)),
        "allowed: the ids are not in increasing order without repeats",
        where + f"got {got!r}",
    )
    accepting = matcher.is_accepting(state)
    check.that(
        accepting is True or accepting is False,
        "is_accepting: must return True or False",
        where + f"got {accepting!r}",
    )
    check.that(
        accepting == grammar.accepts(text),
        f"is_accepting: returned {accepting}, expected {grammar.accepts(text)} for the text so far",
        where + f"got {accepting}, expected {grammar.accepts(text)}\n",
    )
    return got


def follow(ctx, name, vocab, pieces, source=None):
    """Advance through `pieces` (byte strings of tokens), checking every state fully."""
    grammar = Grammar.parse(source if source is not None else GRAMMARS[name])
    matcher = ctx.solution.GrammarMatcher(grammar, vocab)
    state, text = matcher.start(), b""
    check_state(matcher, grammar, name, vocab, state, text)
    for piece in pieces:
        token = vocab.id_of(piece)
        if not grammar.accepts_prefix(text + piece):
            raise RuntimeError(f"authoring error: {piece!r} cannot follow {text!r}")
        state = matcher.advance(state, token)
        text += piece
        check_state(matcher, grammar, name, vocab, state, text)
    return matcher, grammar, state, text


def tokens_of(vocab, text):
    return [vocab.pieces[t] for t in vocab.encode(text)]


@case("the worked example: nested lists, '[1, [true]]' token by token", public=True)
def worked_example(ctx):
    vocab = ToyVocab(size=64, seed=0)
    grammar = Grammar.parse(LISTS)
    matcher = ctx.solution.GrammarMatcher(grammar, vocab)
    values = [6, 8, 9, 12, 16, 17, 20, 26, 27, 32, 34, 37, 45, 46, 49, 51, 52, 54, 56, 60]
    after_value = [48, 50, 62]  # ", " and "]" and ","
    steps = [
        (None, [51], False),
        (b"[", sorted([*values, 29, 50]), False),
        (b"1", sorted([8, 12, 16, 26, 27, 32, 34, 45, 52, 56, *after_value]), False),
        (b",", sorted([*values, 29]), False),
        (b" ", values, False),
        (b"[", sorted([*values, 29, 50]), False),
        (b"tr", [33], False),
        (b"u", [30], False),
        (b"e", after_value, False),
        (b"]", after_value, False),
        (b"]", [], True),
    ]
    state, text = matcher.start(), b""
    for piece, expected, accepting in steps:
        if piece is not None:
            state = matcher.advance(state, vocab.id_of(piece))
            text += piece
        check.equal(matcher.allowed(state), expected, label=f"allowed after {text.decode()!r}")
        check.equal(
            matcher.is_accepting(state), accepting, label=f"is_accepting after {text.decode()!r}"
        )


@case("one token may cross from one rule into the next", public=True)
def tokens_span_rules(ctx):
    # Rejects closure_at_token_end: ', ' is a separator and then optional whitespace,
    # '{"' opens an object and a string, '": ' closes a string and passes a colon.
    vocab = ToyVocab(size=64, seed=0)
    follow(ctx, "nested lists", vocab, tokens_of(vocab, "[1, [true], 20]"))
    follow(ctx, "JSON", vocab, tokens_of(vocab, '{"a": [1, true], "é": "s"} '))


@case("alternatives that begin alike all stay alive until the text decides", public=True)
def alternatives_in_parallel(ctx):
    # Rejects single_stack. 'tr' may be a whole word, the start of 'true', or of
    # 'trust'; a run of letters may stop or go on.
    vocab = ToyVocab(size=64, seed=0)
    name = "alternatives that begin alike"
    for pieces in ([b"tr", b"u", b"s", b"t", b",", b"t", b"r", b"u", b"e", b", ", b"fa", b"l"],
                   [b"tru", b",", b" ", b"true", b",", b"a", b"fa", b"l"]):  # fmt: skip
        follow(ctx, name, vocab, pieces)


@case("left-recursive rules: expr ::= expr '-' term | term", public=True)
def left_recursion(ctx):
    # Rejects left_recursion_loops.
    vocab = ToyVocab(size=64, seed=1)
    follow(ctx, "left-recursive arithmetic", vocab, tokens_of(vocab, "1-20.3-[4-[5]].6"))
    follow(ctx, "left recursion behind an empty rule", vocab, tokens_of(vocab, "  7,,,"))


@case("characters split across tokens: half of '€' is allowed only where '€' could be", public=True)
def split_characters(ctx):
    # Rejects rejects_partial_utf8 and partial_unchecked.
    vocab = ToyVocab(size=64, seed=0)
    name = "non-ASCII characters"
    follow(ctx, name, vocab, [b"\xe2", b"\x82", b"\xac", b"\xf0\x9f", b"\x98\x80", b"\xc3", b"\xa9",
                              b"1", b"\xf0", b"\x9f", b"\x98", b"\x80"])  # fmt: skip
    follow(ctx, name, vocab, [b"{", b"\xe2\x82", b"\xa9", b"\xf0", b"\x9f", b"\x98\x80", b"}"])
    follow(ctx, "a string", vocab, [b'"', b"\xe2\x82", b"\xac", b"\xf0", b"\x98", b"\x80", b"\xac",
                                    b"\xc3", b"\x80", b'"'], source=STRING)  # fmt: skip


@case("a rule that can be empty is needed twice at the same position")
def empty_rules(ctx):
    # Rejects no_nullable_skip: 'ws items ws' with nothing between the brackets.
    vocab = ToyVocab(size=64, seed=2)
    for text in ("[] .", "[ 1 { }]  ", "[12{}]."):
        follow(ctx, "empty rules", vocab, tokens_of(vocab, text))


@case("is_accepting: a finished inner list is not a finished text; half a character is not")
def accepting(ctx):
    # Rejects accepting_any_origin and accepting_mid_character.
    vocab = ToyVocab(size=64, seed=3)
    follow(ctx, "nested lists", vocab, tokens_of(vocab, "[[[]], [1]]"))
    tail = 'root ::= "a" [^x]*'
    follow(
        ctx,
        "any text after 'a'",
        vocab,
        [b"a", b"\xe2", b"\x82", b"\xac", b"\xf0\x9f"],
        source=tail,
    )
    follow(ctx, "bounded repetition", vocab, tokens_of(vocab, "1234-56-78e"))


@case("advance: ValueError for a token that is not allowed; old states stay usable")
def advance_contract(ctx):
    vocab = ToyVocab(size=64, seed=0)
    grammar = Grammar.parse(LISTS)
    matcher = ctx.solution.GrammarMatcher(grammar, vocab)
    start = matcher.start()
    opened = matcher.advance(start, vocab.id_of(b"["))
    for state, text in ((start, b""), (opened, b"[")):
        for token in range(vocab.size):
            piece = vocab.pieces[token]
            if not piece or not grammar.accepts_prefix(text + piece):
                check.raises(
                    ValueError, matcher.advance, state, token,
                    label="advance with a token that is not allowed",
                )  # fmt: skip
    try:
        hash(opened)
    except TypeError:
        check.that(False, "a state is not hashable", f"state: {opened!r}")
    # Branch twice from the same state, then look at the state again.
    one = matcher.advance(opened, vocab.id_of(b"1"))
    other = matcher.advance(opened, vocab.id_of(b"tr"))
    check_state(matcher, grammar, "nested lists", vocab, one, b"[1")
    check_state(matcher, grammar, "nested lists", vocab, other, b"[tr")
    check_state(matcher, grammar, "nested lists", vocab, opened, b"[")
    check_state(matcher, grammar, "nested lists", vocab, start, b"")
    # Two matchers do not share anything.
    second = ctx.solution.GrammarMatcher(Grammar.parse(EXPR), vocab)
    check_state(second, Grammar.parse(EXPR), "left-recursive arithmetic", vocab,
                second.start(), b"")  # fmt: skip
    check_state(matcher, grammar, "nested lists", vocab, one, b"[1")


@case("larger and shuffled vocabularies: every printable ASCII character has a token")
def other_vocabularies(ctx):
    for size, seed in [(100, 4), (132, 5)]:
        vocab = ToyVocab(size=size, seed=seed)
        follow(ctx, "JSON", vocab, tokens_of(vocab, ' {"a": -1.5}'))
        follow(ctx, "non-ASCII characters", vocab, tokens_of(vocab, "é<a A>€"))
        follow(ctx, "alternatives that begin alike", vocab, tokens_of(vocab, "trust,af"))


@case("every byte has a token: bytes that begin no character are never allowed")
def all_bytes(ctx):
    # Rejects overlong_and_surrogates: 0xC0 and 0xF5 start nothing, 0xE0 0x80 would be
    # overlong, 0xED 0xA0 a surrogate, 0xF4 0x90 beyond U+10FFFF.
    vocab = ToyVocab(size=282, seed=6)
    for pieces in ([b'"', b"\xed"], [b'"', b"\xe0"], [b'"', b"\xf4"], [b'"', b"\xf0"],
                   [b'"', b"\xed", b"\x9f"], [b'"', b"\xf4", b"\x8f"]):  # fmt: skip
        grammar = Grammar.parse(STRING)
        matcher = ctx.solution.GrammarMatcher(grammar, vocab)
        state, text = matcher.start(), b""
        for piece in pieces:
            state = matcher.advance(state, vocab.id_of(piece))
            text += piece
        check_state(matcher, grammar, "a string", vocab, state, text)
    follow(ctx, "non-ASCII characters", vocab, [b"{"])


@case("property: random walks through every grammar agree with the oracle")
def random_walks(ctx):
    names = list(GRAMMARS)

    def prop(rng):
        name = names[int(rng.integers(len(names)))]
        vocab = ToyVocab(size=int(rng.choice([64, 64, 90])), seed=int(rng.integers(50)))
        grammar = Grammar.parse(GRAMMARS[name])
        matcher = ctx.solution.GrammarMatcher(grammar, vocab)
        state, text = matcher.start(), b""
        for _ in range(int(rng.integers(3, 12))):
            allowed = check_state(matcher, grammar, name, vocab, state, text, rng=rng)
            if not allowed:
                break
            # Prefer structure over endless digits: half the time, a multi-byte token.
            long = [t for t in allowed if len(vocab.pieces[t]) > 1]
            pool = long if long and rng.random() < 0.5 else allowed
            token = int(pool[int(rng.integers(len(pool)))])
            state = matcher.advance(state, token)
            text += vocab.pieces[token]
            check.that(
                grammar.accepts_prefix(text),
                "advance accepted a token that takes the text outside the language",
                describe(name, vocab, text),
            )
        check_state(matcher, grammar, name, vocab, state, text, rng=rng)

    check.forall(prop, trials=30)
