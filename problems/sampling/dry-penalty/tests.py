"""Tests for dry-penalty.

`model` is the definition from the statement written as plainly as possible: for every
earlier position, count matching tokens backward one by one. It is quadratic and used
only on short histories. Every numeric case compares the solution with it and, as a
check on the reference itself, with the oracle. The speed case times the solution
against the reference on a history that loops, which is the input DRY exists for and
the one where a token-by-token scan is quadratic.
"""

import math

import numpy as np

from learnai.judge import CheckFailure, case, check

FN = "dry_penalties"
FLOAT_MAX_LOG = 88.7228391


def model(history, multiplier, base, allowed_length, breakers, last_n=None):
    window = list(history) if last_n is None else list(history[-last_n:])
    n = len(window)
    longest = {}
    for i in range(n - 1):  # an earlier occurrence ends at i; window[i + 1] followed it
        length = 0
        while (
            length <= i
            and window[i - length] == window[n - 1 - length]
            and window[n - 1 - length] not in breakers
        ):
            length += 1
        candidate = window[i + 1]
        if length >= allowed_length and candidate not in breakers:
            longest[candidate] = max(longest.get(candidate, 0), length)
    max_exponent = int(FLOAT_MAX_LOG / math.log(base)) if base > 1.000001 else 0
    penalties = {}
    for candidate, length in longest.items():
        exponent = length - allowed_length
        if 0 < max_exponent < exponent:
            exponent = max_exponent
        penalties[candidate] = multiplier * base**exponent
    return penalties


def show(history):
    return str(history) if len(history) <= 60 else f"{history[:30]} ... ({len(history)} tokens)"


def run(ctx, history, multiplier, base, allowed_length, breakers, last_n=None, *, want=None):
    """Call the solution; compare with the model (or `want`) and with the oracle."""
    history = [int(t) for t in history]
    breakers = {int(t) for t in breakers}
    call = (
        f"dry_penalties(history, multiplier={multiplier}, base={base}, "
        f"allowed_length={allowed_length}, breakers={sorted(breakers)}, last_n={last_n})\n"
        f"history = {show(history)}\n"
    )
    given = list(history)
    got = ctx.solution.dry_penalties(given, multiplier, base, allowed_length, set(breakers), last_n)
    check.that(given == history, f"{FN}: the history list was modified", call)
    check.that(isinstance(got, dict), f"{FN}: expected a dict, got {type(got).__name__}", call)
    check.that(
        all(type(token) is int for token in got),
        f"{FN}: the keys must be Python ints (token ids)",
        call + f"got {got!r}",
    )
    check.that(
        all(isinstance(p, float) and math.isfinite(p) for p in got.values()),
        f"{FN}: the penalties must be finite Python floats",
        call + f"got {got!r}",
    )
    if want is None:
        want = model(history, multiplier, base, allowed_length, breakers, last_n)
    oracle = ctx.oracle.dry_penalties(
        list(history), multiplier, base, allowed_length, set(breakers), last_n
    )
    if set(oracle) != set(want) or any(
        not math.isclose(oracle[t], want[t], rel_tol=1e-9) for t in want
    ):
        raise RuntimeError(f"authoring error: the oracle gives {oracle}, the model {want}\n{call}")
    detail = call + f"expected {dict(sorted(want.items()))}\ngot      {dict(sorted(got.items()))}"
    extra, missing = sorted(set(got) - set(want)), sorted(set(want) - set(got))
    if extra or missing:
        problems = []
        if extra:
            problems.append(f"{len(extra)} token(s) are penalized that should not be")
        if missing:
            problems.append(f"{len(missing)} token(s) that extend a repeat have no penalty")
        raise CheckFailure(f"{FN}: " + " and ".join(problems), detail)
    wrong = [t for t in want if not math.isclose(got[t], want[t], rel_tol=1e-9)]
    if wrong:
        raise CheckFailure(
            f"{FN}: {len(wrong)} of {len(want)} penalties have the wrong size "
            f"(for example the penalty of token {wrong[0]}, {got[wrong[0]]:.6g})",
            detail,
        )
    return got


A, B, C, Y = 1, 2, 3, 9
EXAMPLE = [A, B, C, C, B, C, Y, A, B, C]


@case("the worked example: a b c c b c y a b c", public=True)
def worked_example(ctx):
    got = run(ctx, EXAMPLE, 0.8, 1.75, 2, set(), want={C: 0.8 * 1.75, Y: 0.8})
    check.equal(sorted(got), [C, Y], label="penalized tokens")
    # allowed_length = 1: the one-token match `c` before `b` now counts as well.
    run(ctx, EXAMPLE, 0.8, 1.75, 1, set(), want={C: 0.8 * 1.75**2, Y: 0.8 * 1.75, B: 0.8})
    run(ctx, EXAMPLE, 0.8, 1.75, 4, set(), want={})


@case("sequence breakers: a match stops at a breaker, and a breaker is never penalized",
      public=True)  # fmt: skip
def breakers_example(ctx):
    # b is a breaker: only `c` follows the last breaker, so no match reaches length 2.
    run(ctx, EXAMPLE, 0.8, 1.75, 2, {B}, want={})
    # a is a breaker: matches are capped at the two tokens `b c` after it.
    run(ctx, EXAMPLE, 0.8, 1.75, 2, {A}, want={C: 0.8, Y: 0.8})
    # y is a breaker: it would extend `b c`, but breakers are not penalized.
    run(ctx, EXAMPLE, 0.8, 1.75, 2, {Y}, want={C: 0.8 * 1.75})
    # The last token is a breaker: nothing can match.
    run(ctx, EXAMPLE, 0.8, 1.75, 1, {C}, want={})


@case("a loop: the penalty grows by the factor `base` with every repeated token", public=True)
def growing_loop(ctx):
    verse = [11, 12, 13, 14, 15]
    history = [7, 8, *verse, *verse]
    for extra in range(1, 6):
        # After `extra` tokens of the third verse the match is 5 + extra tokens long.
        got = run(ctx, history + verse[:extra], 1.0, 2.0, 2, set())
        check.equal(sorted(got), [verse[extra % 5]], label="penalized tokens")


@case("last_n: only the last tokens of the history are searched", public=True)
def lookback(ctx):
    history = [5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7]
    run(ctx, history, 0.5, 2.0, 2, set(), want={8: 1.0})
    run(ctx, history, 0.5, 2.0, 2, set(), 10, want={8: 0.5})  # `5` fell out of the window
    run(ctx, history, 0.5, 2.0, 2, set(), 9, want={})  # the window starts at `7 8`
    run(ctx, history, 0.5, 2.0, 2, set(), 50, want={8: 1.0})


def looping_history(rng, n, period):
    block = [int(t) for t in rng.integers(100, 1000, size=period)]
    head = [int(t) for t in rng.integers(100, 1000, size=37)]
    return head + (block * (n // period + 1))[:n]


@case("speed: a looping history of 20,000 tokens, within 5x of the linear-time reference",
      public=True, time_limit_s=30)  # fmt: skip
def linear_time(ctx):
    history = looping_history(ctx.rng, 20_000, 20)
    short = history[:600]
    run(ctx, short, 0.8, 1.75, 2, {5})  # correct on the same kind of input first
    ctx.check_speed(FN, history, 0.8, 1.75, 2, {5}, max_ratio=5.0, rounds=3, warmup=1)


@case("the longest match wins when a token follows several earlier occurrences")
def longest_per_token(ctx):
    # `4` follows `1 2 3` (match 3) early and `3` alone (match 1) later.
    run(ctx, [1, 2, 3, 4, 9, 3, 4, 8, 1, 2, 3], 1.0, 3.0, 1, set(), want={4: 9.0})
    # The same with the short occurrence first.
    run(ctx, [3, 4, 9, 1, 2, 3, 4, 8, 1, 2, 3], 1.0, 3.0, 1, set(), want={4: 9.0})
    # Two candidates with different lengths, and one below allowed_length.
    run(ctx, [2, 3, 5, 0, 1, 2, 3, 6, 0, 3, 7, 1, 2, 3], 2.0, 1.5, 2, set())


@case("a match of exactly allowed_length tokens is penalized by `multiplier`; shorter is not")
def exact_allowed_length(ctx):
    history = [1, 2, 3, 4, 0, 2, 3, 4]
    run(ctx, history, 0.7, 1.75, 3, set(), want={0: 0.7})
    run(ctx, history, 0.7, 1.75, 4, set(), want={})
    run(ctx, [4, 4], 0.7, 1.75, 1, set(), want={4: 0.7})
    run(ctx, [1, 2, 1], 0.7, 1.75, 1, set(), want={2: 0.7})


@case("a repeat that overlaps itself: runs of one token and a period of two")
def self_overlap(ctx):
    run(ctx, [6] * 7, 1.0, 2.0, 2, set(), want={6: 2.0**4})
    run(ctx, [1, 2] * 5, 1.0, 2.0, 2, set(), want={1: 2.0**6})
    run(ctx, [3, 1, 2, 1, 2, 1], 1.0, 2.0, 1, set())


@case("breakers inside the history: before, between, and inside the repeated run")
def breakers_inside(ctx):
    history = [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    run(ctx, history, 1.0, 2.0, 2, {0}, want={5: 4.0})  # 0 precedes the suffix: no effect
    run(ctx, history, 1.0, 2.0, 2, {2}, want={5: 1.0})  # capped at `3 4`
    run(ctx, history, 1.0, 2.0, 2, {3}, want={})  # capped at `4`, below allowed_length
    run(ctx, history, 1.0, 2.0, 1, {3}, want={5: 1.0})
    run(ctx, history, 1.0, 2.0, 2, {5}, want={})  # the only candidate is a breaker
    run(ctx, [1, 2, 9, 1, 2, 7, 1, 2], 1.0, 2.0, 2, {9}, want={7: 1.0})
    run(ctx, [1, 2, 3], 1.0, 2.0, 1, {1, 2, 3}, want={})


@case("a long loop: the exponent is clamped where base ** exponent would overflow float32")
def exponent_clamp(ctx):
    history = [4] * 3000
    clamp = int(FLOAT_MAX_LOG / math.log(1.75))
    run(ctx, history, 0.8, 1.75, 2, set(), want={4: 0.8 * 1.75**clamp})
    run(
        ctx,
        [1, 2, 3] * 200,
        2.0,
        3.0,
        2,
        set(),
        want={1: 2.0 * 3.0 ** int(FLOAT_MAX_LOG / math.log(3.0))},
    )
    # At the clamp and one below it: no clamping yet.
    for length in (clamp + 1, clamp + 2, clamp + 3):
        run(ctx, [4] * (length + 1), 0.8, 1.75, 2, set())
    # base = 1: every penalty is the multiplier, however long the match.
    run(ctx, history, 0.8, 1.0, 2, set(), want={4: 0.8})
    run(ctx, [4] * 400, 1.0, 1.0000005, 2, set())


@case("empty and tiny histories, last_n = 1, and a window that cuts a repeat")
def edges(ctx):
    run(ctx, [], 0.8, 1.75, 2, set(), want={})
    run(ctx, [5], 0.8, 1.75, 1, set(), want={})
    run(ctx, [5, 5, 5], 0.8, 1.75, 1, set(), 1, want={})
    run(ctx, [5, 5, 5], 0.8, 1.75, 1, set(), 2, want={5: 0.8})
    run(ctx, [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3], 1.0, 2.0, 1, set(), 6)
    run(ctx, [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3], 1.0, 2.0, 1, {9}, 8)


def random_history(rng):
    """Short histories over a tiny alphabet, with loops spliced in: many nested repeats."""
    alphabet = int(rng.integers(2, 6))
    pieces = []
    for _ in range(int(rng.integers(1, 5))):
        block = [int(t) for t in rng.integers(0, alphabet, size=int(rng.integers(1, 9)))]
        pieces += block * int(rng.integers(1, 5))
        pieces += [int(t) for t in rng.integers(0, alphabet, size=int(rng.integers(0, 4)))]
    return pieces, alphabet


@case("random histories over two to five tokens: nested and overlapping repeats")
def random_small_alphabet(ctx):
    for _ in range(300):
        history, alphabet = random_history(ctx.rng)
        allowed = int(ctx.rng.integers(1, 5))
        run(ctx, history, 0.8, 1.75, allowed, set())
        if alphabet > 2:
            breakers = {int(t) for t in ctx.rng.integers(0, alphabet, size=1)}
            run(ctx, history, 1.3, 1.2, allowed, breakers)


@case("random histories with breakers and random lookback windows")
def random_windows(ctx):
    for _ in range(200):
        history, alphabet = random_history(ctx.rng)
        history += [alphabet + 1] if ctx.rng.random() < 0.2 else []
        breakers = {alphabet + 1} | {int(t) for t in ctx.rng.integers(0, 2 * alphabet, size=1)}
        last_n = int(ctx.rng.integers(1, len(history) + 3))
        run(ctx, history, 0.5, 2.5, int(ctx.rng.integers(1, 4)), breakers, last_n)


@case("a looping history of 600 tokens with a breaker in the middle of the loop")
def long_loop_with_breaker(ctx):
    history = looping_history(np.random.default_rng(ctx.seed), 600, 24)
    run(ctx, history, 0.8, 1.75, 2, {history[-9]})
    run(ctx, history, 0.8, 1.75, 2, set(), 100)
    run(ctx, history, 0.8, 1.02, 2, set())
