"""Tests for dfa-token-mask.

The expected masks are computed here from the definition, by walking every token's
bytes through the automaton with its uncounted internals, so they do not depend on the
reference. The property case checks the meaning of a mask: whatever allowed tokens are
chosen, the text so far is a prefix of a string of the language.
"""

from learnai.judge import case, check
from learnai.kit import DEAD, CharDFA, ToyVocab

OBJECT = r'\{"[a-z]+": (true|false|null|-?\d+)\}'
NUMBER = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?"
LIST = r"\[(true|false)(, (true|false))*\]"
STRING = r'"[^"]*"'
EMOJI = r"(é|€|😀)+"


def walk(dfa, state, piece):
    """Uncounted walk, for the tests only."""
    for byte in piece:
        state = dfa._next(state, byte)
    return state


def moves(dfa, state, vocab):
    """{token: next state} for every allowed token, from the definition."""
    out = {}
    for token, piece in enumerate(vocab.pieces):
        after = walk(dfa, state, piece) if piece else DEAD
        if after != DEAD:
            out[token] = after
    return out


def show(vocab, tokens):
    return ", ".join(f"{t}={vocab.pieces[t]!r}" for t in tokens)


def check_state(ctx, dfa, state, vocab, what):
    """allowed_tokens and advance in one state, against the definition."""
    expected = moves(dfa, state, vocab)
    label = f"allowed_tokens in state {state} of {what}"
    got = ctx.call("allowed_tokens", dfa, state, vocab)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{label}: must return a list of Python ints",
        f"got {got!r}",
    )
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    known = [token for token in extra if 0 <= token < vocab.size]
    unknown = [token for token in extra if not 0 <= token < vocab.size]
    check.that(
        not missing and not extra,
        f"{label}: wrong set of tokens",
        f"allowed but missing from your list: {show(vocab, missing) or 'none'}\n"
        f"in your list but not allowed: {show(vocab, known) or 'none'}\n"
        + (f"in your list but not token ids at all: {unknown}\n" if unknown else ""),
    )
    check.equal(got, sorted(expected), label=label)
    for token, after in expected.items():
        got_state = ctx.call("advance", dfa, state, token, vocab)
        check.equal(
            got_state,
            after,
            label=f"advance from state {state} of {what} with token {vocab.pieces[token]!r}",
        )


def check_all_states(ctx, pattern, vocab):
    dfa = CharDFA.from_regex(pattern)
    for state in dfa.states:
        check_state(ctx, dfa, state, vocab, f"the DFA of {pattern}")
    return dfa


@case("the worked example: (true|null) with ToyVocab(64, seed=0)", public=True)
def worked_example(ctx):
    vocab = ToyVocab(size=64, seed=0)
    dfa = CharDFA.from_regex(r"(true|null)")
    start = ctx.call("allowed_tokens", dfa, 0, vocab)
    check.equal(start, [6, 19, 40, 46, 47, 49, 53, 54], label="allowed_tokens in state 0")
    check.equal(ctx.call("advance", dfa, 0, 46, vocab), 4, label="advance(dfa, 0, 46 'tr')")
    check.equal(ctx.call("allowed_tokens", dfa, 4, vocab), [33], label="allowed_tokens in state 4")
    check.equal(ctx.call("advance", dfa, 4, 33, vocab), 6, label="advance(dfa, 4, 33 'u')")
    check.equal(ctx.call("allowed_tokens", dfa, 6, vocab), [30], label="allowed_tokens in state 6")
    check.equal(ctx.call("advance", dfa, 6, 30, vocab), 7, label="advance(dfa, 6, 30 'e')")
    check.equal(ctx.call("allowed_tokens", dfa, 7, vocab), [], label="allowed_tokens in state 7")
    check.equal(ctx.call("allowed_tokens", dfa, 1, vocab), [33], label="allowed_tokens in state 1")
    check.equal(ctx.call("advance", dfa, 0, 47, vocab), 7, label="advance(dfa, 0, 47 'null')")


@case("tokens that span several transitions and end in non-accepting states", public=True)
def every_state_of_an_object(ctx):
    # Rejects requires_accepting (nearly no token ends a complete object),
    # first_byte_only ('": ' right after the opening quote), and ignores_state.
    check_all_states(ctx, OBJECT, ToyVocab(size=64, seed=0))


@case("special tokens are never allowed, not even in an accepting state", public=True)
def specials_never_allowed(ctx):
    # Rejects specials_allowed.
    vocab = ToyVocab(size=64, seed=1)
    for pattern in (r"[0-9]*", r"(true|false)?"):
        dfa = check_all_states(ctx, pattern, vocab)
        for state in dfa.states:
            got = ctx.call("allowed_tokens", dfa, state, vocab)
            specials = [token for token in got if vocab.is_special(token)]
            check.that(
                not specials,
                f"allowed_tokens: a special token is allowed in state {state} of {pattern}",
                f"special tokens in the list: {specials}",
            )


@case("advance raises ValueError for a token that is not allowed", public=True)
def advance_rejects(ctx):
    # Rejects advance_returns_dead.
    vocab = ToyVocab(size=64, seed=0)
    dfa = CharDFA.from_regex(r"(true|null)")
    for state, piece in [(0, b"fa"), (0, b"u"), (4, b"tr"), (1, b"nu"), (7, b"e"), (2, b"ru")]:
        if piece in vocab.pieces:
            token = vocab.id_of(piece)
            check.raises(
                ValueError, ctx.call, "advance", dfa, state, token, vocab,
                label=f"advance from state {state} with token {piece!r}",
            )  # fmt: skip
    for state in (0, 4, 7):
        check.raises(
            ValueError, ctx.call, "advance", dfa, state, vocab.eos_id, vocab,
            label=f"advance from state {state} with the EOS token",
        )  # fmt: skip


@case("characters split across tokens: é, €, and 😀 inside a string and on their own")
def multibyte_characters(ctx):
    # Rejects rejects_partial_utf8: b'\xe2\x82' is allowed where '€' may start, and
    # b'\xac' only right after it.
    for seed in (0, 2):
        vocab = ToyVocab(size=64, seed=seed)
        check_all_states(ctx, EMOJI, vocab)
        check_all_states(ctx, STRING, vocab)
        check_all_states(ctx, r"[^0-9]+", vocab)


@case("numbers, lists, and any text, with shuffled and larger vocabularies")
def more_patterns(ctx):
    for size, seed in [(64, 3), (100, 4), (282, 5)]:
        vocab = ToyVocab(size=size, seed=seed)
        for pattern in (NUMBER, LIST, OBJECT, r".*", r"(true|null)", r'\{"\w+": "[^"\\]*"\}'):
            check_all_states(ctx, pattern, vocab)


@case("a hand-built DFA whose states are not numbered from 0")
def hand_built(ctx):
    vocab = ToyVocab(size=64, seed=6)
    # 5 --t--> 9 --r--> 12 --u--> 5, 12 --e--> 30 (accepting), 30 --","--> 5
    dfa = CharDFA(
        {5: {"t": 9}, 9: {"r": 12}, 12: {"u": 5, "e": 30}, 30: {",": 5, " ": 30}},
        start=5,
        accepting=[30],
    )
    for state in dfa.states:
        check_state(ctx, dfa, state, vocab, "a hand-built DFA")


@case("property: any sequence of allowed tokens spells a prefix of the language")
def allowed_sequences_are_prefixes(ctx):
    patterns = [OBJECT, NUMBER, LIST, STRING, EMOJI, r"(true|false|null)(, (true|false|null))*"]

    def prop(rng):
        pattern = patterns[int(rng.integers(len(patterns)))]
        vocab = ToyVocab(size=int(rng.choice([64, 80, 282])), seed=int(rng.integers(100)))
        dfa = CharDFA.from_regex(pattern)
        state, text, tokens = dfa.start, b"", []
        for _ in range(int(rng.integers(1, 25))):
            allowed = ctx.call("allowed_tokens", dfa, state, vocab)
            expected = sorted(moves(dfa, state, vocab))
            detail = (
                f"pattern: {pattern}\nvocabulary: {vocab!r}\ntokens so far: {tokens}\n"
                f"text so far: {text!r}\nDFA state: {state}\n"
            )
            check.that(
                allowed == expected,
                "allowed_tokens: wrong tokens after a sequence of allowed tokens",
                detail + f"got      {allowed!r}\nexpected {expected!r}\n",
            )
            if not allowed:
                break
            token = int(allowed[int(rng.integers(len(allowed)))])
            state = ctx.call("advance", dfa, state, token, vocab)
            tokens.append(token)
            text += vocab.pieces[token]
            detail += f"then token {token} = {vocab.pieces[token]!r}, new state {state!r}\n"
            check.that(
                dfa.accepts_prefix(text),
                "an allowed token took the text outside the language",
                detail,
            )
            check.that(
                state == dfa._run(text),
                "advance: the returned state is not the state of the text so far",
                detail,
            )

    check.forall(prop, trials=40)
