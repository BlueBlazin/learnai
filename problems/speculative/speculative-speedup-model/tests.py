"""Tests for speculative-speedup-model.

Expected values are computed here in exact rational arithmetic from the series
1 + alpha + ... + alpha**k (never from the closed form), so they are right to the last
bit for every alpha, and compared to a relative tolerance of 1e-9. `best_k` inputs are
used only when the exact maximum is unique by a relative margin of 1e-6, as the
statement promises.
"""

from fractions import Fraction

from learnai.judge import case, check

RTOL = 1e-9
MARGIN = Fraction(1, 10**6)


def exact_tokens(alpha, k):
    a = Fraction(alpha)  # the float's exact value
    return sum(a**i for i in range(k + 1))


def exact_speedup(alpha, k, c):
    return exact_tokens(alpha, k) / (k * Fraction(c) + 1)


def exact_best_k(alpha, c, k_max):
    """(best k, True if the maximum is unique by the promised margin)."""
    values = [exact_speedup(alpha, k, c) for k in range(k_max + 1)]
    best = max(range(k_max + 1), key=values.__getitem__)
    clear = all(values[best] >= v * (1 + MARGIN) for k, v in enumerate(values) if k != best)
    return best, clear


def check_float(ctx, name, args, exact, literal=None):
    label = f"{name}{args}"
    got = ctx.call(name, *args)
    check.that(
        type(got) is float, f"{label}: the result must be a Python float", f"got: {got!r}"
    )
    want = float(exact)
    if literal is not None and abs(literal - want) > 1e-12 * want:  # an authoring error
        raise RuntimeError(f"{label}: the literal in the statement is wrong")
    check.allclose(got, want, rtol=RTOL, atol=0.0, label=label)
    # The oracle must be within the tolerance too, or the tolerance is unfair.
    check.allclose(ctx.expected(name, *args), want, rtol=RTOL / 100, atol=0.0, label="oracle")


def check_tokens(ctx, alpha, k, literal=None):
    check_float(ctx, "expected_tokens", (alpha, k), exact_tokens(alpha, k), literal)


def check_speedup(ctx, alpha, k, c, literal=None):
    check_float(ctx, "speedup", (alpha, k, c), exact_speedup(alpha, k, c), literal)


def check_best_k(ctx, alpha, c, k_max, literal=None):
    label = f"best_k({alpha}, {c}, {k_max})"
    want, clear = exact_best_k(alpha, c, k_max)
    if not clear or (literal is not None and literal != want):  # an authoring error
        raise RuntimeError(f"{label}: the maximum is not clear, or the literal is wrong")
    got = ctx.call("best_k", alpha, c, k_max)
    check.that(type(got) is int, f"{label}: the result must be a Python int", f"got: {got!r}")
    check.that(
        got == want,
        f"{label}: wrong draft length",
        f"expected {want}, got {got}\nspeedup at the expected k: "
        f"{float(exact_speedup(alpha, want, c)):.6f}"
        + (
            f"\nspeedup at your k: {float(exact_speedup(alpha, got, c)):.6f}"
            if 0 <= got <= 64
            else ""
        ),
    )


@case("the worked example: alpha = 0.8, k = 4, c = 0.05, and the three best_k calls", public=True)
def worked_example(ctx):
    check_tokens(ctx, 0.8, 4, literal=3.3616)
    check_speedup(ctx, 0.8, 4, 0.05, literal=2.8013333333333335)
    check_best_k(ctx, 0.8, 0.05, 20, literal=8)
    check_best_k(ctx, 0.8, 0.3, 20, literal=3)
    check_best_k(ctx, 0.3, 0.5, 20, literal=0)
    check_tokens(ctx, 1.0, 4, literal=5.0)
    check_speedup(ctx, 1.0, 4, 0.05, literal=4.166666666666667)


@case("the table of the statement: alpha = 0.8 for k from 0 to 12, c = 0.05 and 0.3", public=True)
def statement_table(ctx):
    # Rejects no_bonus_token, cost_without_target_call, and extra_draft_call row by row.
    for k in range(13):
        check_tokens(ctx, 0.8, k)
        check_speedup(ctx, 0.8, k, 0.05)
        check_speedup(ctx, 0.8, k, 0.3)


@case("alpha = 1.0: every draft is accepted, so a round yields k + 1 tokens", public=True)
def perfect_draft(ctx):
    # Rejects zero_division_at_one and epsilon_at_one.
    for k in (0, 1, 7, 64):
        check_tokens(ctx, 1.0, k)
        check_speedup(ctx, 1.0, k, 0.25)
    check_best_k(ctx, 1.0, 0.05, 20, literal=20)
    check_best_k(ctx, 1.0, 0.0, 64, literal=64)


@case("speculation that does not pay: best_k is 0 when the draft is slow or often wrong",
      public=True)
def declines(ctx):
    # Rejects never_declines. Speculation can win only if alpha > c.
    check_best_k(ctx, 0.3, 0.5, 20, literal=0)
    check_best_k(ctx, 0.9, 1.5, 8, literal=0)
    check_best_k(ctx, 0.0, 0.01, 8, literal=0)
    check_best_k(ctx, 0.5, 0.2, 0, literal=0)
    check_best_k(ctx, 0.61, 0.6, 30, literal=1)


@case("k = 0 and alpha = 0: one token per round, speedup 1 / (k c + 1)")
def degenerate(ctx):
    for alpha in (0.0, 0.37, 0.999, 1.0):
        check_tokens(ctx, alpha, 0)
        check_speedup(ctx, alpha, 0, 0.4)
    for k in (1, 5, 64):
        check_tokens(ctx, 0.0, k)
        check_speedup(ctx, 0.0, k, 0.1)


@case("a free draft (c = 0) and a draft slower than the target (c > 1)")
def cost_extremes(ctx):
    for alpha, k in [(0.6, 3), (0.95, 20), (1.0, 9), (0.2, 64)]:
        check_speedup(ctx, alpha, k, 0.0)
        check_speedup(ctx, alpha, k, 1.0)
        check_speedup(ctx, alpha, k, 4.0)
    check_best_k(ctx, 0.9, 0.0, 20, literal=20)
    check_best_k(ctx, 0.9, 0.0, 3, literal=3)


@case("acceptance rates close to 1 (up to 0.999) with long drafts")
def near_one(ctx):
    for alpha in (0.9, 0.99, 0.995, 0.999):
        for k in (1, 2, 16, 64):
            check_tokens(ctx, alpha, k)
            check_speedup(ctx, alpha, k, 0.02)


@case("best_k is limited by k_max, and the peak lies strictly inside the range otherwise")
def best_k_range(ctx):
    # Rejects maximizes_tokens (always k_max) on the interior peaks.
    check_best_k(ctx, 0.8, 0.05, 5, literal=5)
    check_best_k(ctx, 0.8, 0.05, 64, literal=8)
    check_best_k(ctx, 0.95, 0.02, 64)
    check_best_k(ctx, 0.5, 0.1, 64)
    check_best_k(ctx, 0.99, 0.01, 64)


@case("random acceptance rates, draft lengths, and cost ratios")
def random_sweep(ctx):
    rng = ctx.rng
    searched = 0
    for _ in range(60):
        alpha = float(rng.choice([rng.uniform(0, 0.999), rng.uniform(0.5, 0.999), 1.0]))
        k = int(rng.integers(0, 65))
        c = float(rng.choice([rng.uniform(0, 1), 10 ** rng.uniform(-3, 0.6), 0.0]))
        check_tokens(ctx, alpha, k)
        check_speedup(ctx, alpha, k, c)
        k_max = int(rng.integers(0, 65))
        if exact_best_k(alpha, c, k_max)[1]:
            check_best_k(ctx, alpha, c, k_max)
            searched += 1
    if searched < 40:  # an authoring error
        raise RuntimeError("too few best_k inputs have a clear maximum")
