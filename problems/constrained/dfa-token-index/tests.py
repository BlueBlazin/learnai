"""Tests for dfa-token-index.

Correctness: for every state of the automaton, the tokens the index allows and the
states it leads to are compared with the definition (every token's bytes walked
through the automaton, with its uncounted internals, here in the tests).

Cost, through `dfa.counters.steps`:

* building may take at most one step per *live pair*: a state `q` and a distinct
  non-empty prefix `p` of some token's bytes such that `p` without its last byte
  survives from `q` (`step_budget` below). A vocabulary trie walked once per state
  takes exactly that many; walking every token separately takes more.
* after building, queries take no steps at all.

The index is opaque, and it can be a large nested dict, so the three functions are
called directly on the solution module rather than through `ctx.call`, which would
copy the index on every query.
"""

from learnai.judge import case, check
from learnai.kit import DEAD, CharDFA, ToyVocab

OBJECT = r'\{"[a-z]+": (true|false|null|-?\d+)\}'
NUMBER = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?"
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


def step_budget(dfa, vocab):
    """The number of live (state, prefix) pairs: the statement's limit for build_index."""
    prefixes = {piece[:n] for piece in vocab.pieces for n in range(1, len(piece) + 1)}
    return sum(
        1 for state in dfa.states for prefix in prefixes if walk(dfa, state, prefix[:-1]) != DEAD
    )


def naive_steps(dfa, vocab):
    """What walking every token separately from every state costs."""
    twin = dfa.with_framework(None)
    for state in twin.states:
        for piece in vocab.pieces:
            twin.walk(state, piece)
    return twin.counters.steps


def show(vocab, tokens):
    known = [t for t in tokens if isinstance(t, int) and 0 <= t < vocab.size]
    other = [t for t in tokens if t not in known]
    text = ", ".join(f"{t}={vocab.pieces[t]!r}" for t in known)
    return (text + (f" and non-token values {other!r}" if other else "")) or "none"


def build(ctx, dfa, vocab):
    """Build the solution's index and check the step budget. Returns the index."""
    index = ctx.solution.build_index(dfa, vocab)
    check.cost(dfa.counters, "steps", at_most=step_budget(dfa, vocab), unit="steps")
    return index


def check_queries(ctx, index, dfa, vocab, what):
    """Every state: the allowed list, every advance, some refusals; then zero steps."""
    before = dfa.counters.steps
    for state in dfa.states:
        expected = moves(dfa, state, vocab)
        label = f"allowed_from_index in state {state} of {what}"
        got = ctx.solution.allowed_from_index(index, state)
        check.that(
            isinstance(got, list) and all(type(token) is int for token in got),
            f"{label}: must return a list of Python ints",
            f"got {got!r}",
        )
        check.that(
            set(got) == set(expected),
            f"{label}: wrong set of tokens",
            f"allowed but missing from your list: {show(vocab, sorted(set(expected) - set(got)))}\n"
            f"in your list but not allowed: {show(vocab, sorted(set(got) - set(expected)))}\n",
        )
        check.equal(got, sorted(expected), label=label)
        for token, after in expected.items():
            check.equal(
                ctx.solution.advance_from_index(index, state, token),
                after,
                label=f"advance_from_index from state {state} of {what} "
                f"with token {vocab.pieces[token]!r}",
            )
        refused = [token for token in range(vocab.size) if token not in expected]
        for token in [*refused[:5], *refused[-12:]]:  # the five specials and a dozen others
            check.raises(
                ValueError, ctx.solution.advance_from_index, index, state, token,
                label=f"advance_from_index from state {state} of {what} "
                f"with the disallowed token {token} ({vocab.pieces[token]!r})",
            )  # fmt: skip
    check.cost(
        {"steps_after_build": dfa.counters.steps - before}, "steps_after_build", exactly=0
    )


def check_pattern(ctx, pattern, vocab):
    dfa = CharDFA.from_regex(pattern)
    index = build(ctx, dfa, vocab)
    check_queries(ctx, index, dfa, vocab, f"the DFA of {pattern}")
    return dfa, index


@case("the worked example: (true|null), 310 steps to build, none afterward", public=True)
def worked_example(ctx):
    vocab = ToyVocab(size=64, seed=0)
    dfa = CharDFA.from_regex(r"(true|null)")
    index = ctx.solution.build_index(dfa, vocab)
    built = dfa.counters.steps
    table = {
        0: {6: 2, 19: 1, 40: 5, 46: 4, 47: 7, 49: 7, 53: 3, 54: 6},
        1: {33: 3},
        2: {43: 4},
        3: {18: 5},
        4: {33: 6},
        5: {18: 7},
        6: {30: 7},
        7: {},
    }
    for state, expected in table.items():
        got = ctx.solution.allowed_from_index(index, state)
        check.equal(got, sorted(expected), label=f"allowed_from_index(index, {state})")
        for token, after in expected.items():
            got = ctx.solution.advance_from_index(index, state, token)
            check.equal(got, after, label=f"advance_from_index(index, {state}, {token})")
    check.raises(
        ValueError, ctx.solution.advance_from_index, index, 1, 53,
        label="advance_from_index(index, 1, 53)",
    )  # fmt: skip
    check.cost(
        {"steps_after_build": dfa.counters.steps - built}, "steps_after_build", exactly=0
    )
    # 44 live pairs in state 0 (38 first bytes, then tr, tru, true, nu, nul, null) and
    # 38 in each of the other seven states. Walking each token separately takes 484.
    check.cost({"steps": built}, "steps", at_most=310, unit="steps")


@case("a JSON-like object: every state against the definition, within the budget", public=True)
def object_pattern(ctx):
    # Rejects keyed_by_token_only, dead_states_stored, and interior_tokens_lost.
    check_pattern(ctx, OBJECT, ToyVocab(size=64, seed=0))


@case("queries after the build never step the DFA", public=True)
def queries_are_free(ctx):
    # Rejects lazy_walk. A decoding run: 200 masks and advances along a random path.
    vocab = ToyVocab(size=64, seed=1)
    dfa = CharDFA.from_regex(r"\[(true|false|null|-?\d+)(, (true|false|null|-?\d+))*\]")
    index = ctx.solution.build_index(dfa, vocab)
    before = dfa.counters.steps
    state = dfa.start
    for _ in range(200):
        allowed = ctx.solution.allowed_from_index(index, state)
        check.equal(allowed, sorted(moves(dfa, state, vocab)), label="allowed_from_index")
        if not allowed:
            state = dfa.start
            continue
        token = allowed[int(ctx.rng.integers(len(allowed)))]
        after = ctx.solution.advance_from_index(index, state, token)
        check.equal(after, walk(dfa, state, vocab.pieces[token]), label="advance_from_index")
        state = after
    check.cost(
        {"steps_after_build": dfa.counters.steps - before}, "steps_after_build", exactly=0
    )


@case("the build shares the steps of common prefixes: any text, where nothing dies early")
def permissive_pattern(ctx):
    # Rejects naive_build by the widest margin: with .* every token survives, so a
    # token-by-token walk pays for every byte of every token.
    vocab = ToyVocab(size=64, seed=2)
    dfa, _ = check_pattern(ctx, r".*", vocab)
    if naive_steps(dfa, vocab) <= step_budget(dfa, vocab):  # an authoring error
        raise RuntimeError("the budget does not separate the naive build")


@case("a restrictive pattern: most prefixes die at once and must not be walked further")
def restrictive_pattern(ctx):
    # Rejects no_pruning and naive_build.
    for seed in (0, 3):
        check_pattern(ctx, r"(true|null)", ToyVocab(size=64, seed=seed))
        check_pattern(ctx, NUMBER, ToyVocab(size=64, seed=seed))


@case("special tokens are in no state's list, accepting or not")
def specials_excluded(ctx):
    # Rejects specials_at_root.
    vocab = ToyVocab(size=64, seed=4)
    for pattern in (r"[0-9]*", r"(true|false)?"):
        dfa, index = check_pattern(ctx, pattern, vocab)
        for state in dfa.states:
            got = ctx.solution.allowed_from_index(index, state)
            check.that(
                not any(vocab.is_special(token) for token in got),
                f"allowed_from_index: a special token is allowed in state {state} of {pattern}",
                f"got {got!r}",
            )


@case("characters split across tokens, and larger vocabularies")
def multibyte_and_large(ctx):
    check_pattern(ctx, EMOJI, ToyVocab(size=64, seed=5))
    check_pattern(ctx, STRING, ToyVocab(size=64, seed=6))
    check_pattern(ctx, STRING, ToyVocab(size=282, seed=7))
    check_pattern(ctx, OBJECT, ToyVocab(size=120, seed=8))
    check_pattern(ctx, r'\{"\w+": "[^"\\]*"\}', ToyVocab(size=282, seed=9))


@case("a hand-built DFA whose states are not numbered from 0, and two indexes at once")
def hand_built(ctx):
    vocab = ToyVocab(size=64, seed=10)
    # 5 --t--> 9 --r--> 12 --u--> 5, 12 --e--> 30 (accepting), 30 --","--> 5
    first = CharDFA(
        {5: {"t": 9}, 9: {"r": 12}, 12: {"u": 5, "e": 30}, 30: {",": 5, " ": 30}},
        start=5,
        accepting=[30],
    )
    second = CharDFA.from_regex(r"(true|false)")
    index_first = build(ctx, first, vocab)
    index_second = build(ctx, second, vocab)
    check_queries(ctx, index_first, first, vocab, "a hand-built DFA")
    check_queries(ctx, index_second, second, vocab, "the DFA of (true|false)")
    check_queries(ctx, index_first, first, vocab, "a hand-built DFA")
