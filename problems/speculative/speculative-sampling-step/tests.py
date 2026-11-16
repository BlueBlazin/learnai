"""Tests for speculative-sampling-step.

The numeric cases pin down how the two uniform numbers are used, by comparing single
calls with the oracle. Their inputs stay clear of every decision boundary (`decisive`
checks it in float64), so float32 rounding can never decide a call.

The statistical cases check the theorem: when the draft token is drawn from q and the
uniforms are independent, the emitted token is distributed exactly as p. There are five
statistical checks in this file, each with the default delta of 1e-10, which keeps the
false-failure rate of a correct solution below 1e-9.
"""

import numpy as np

from learnai.judge import case, check

FN = "speculative_step"
MARGIN = 1e-3

# The worked example of the statement. Residual max(0, p - q) = [0, 0.2, 0, 0.2].
P4 = np.array([0.1, 0.4, 0.2, 0.3], dtype=np.float32)
Q4 = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float32)


def step(ctx, p, q, x, u_accept, u_resample):
    """Call the solution and return `(token, accepted)` as a Python int and bool."""
    result = ctx.call(FN, p, q, int(x), float(u_accept), float(u_resample))
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{FN}: expected a (token, accepted) pair, got {type(result).__name__}",
    )
    token, accepted = (np.asarray(value) for value in result)
    check.that(
        token.size == 1 and token.dtype.kind in "iu",
        f"{FN}: the token must be a single integer, got dtype {token.dtype}, shape {token.shape}",
    )
    check.that(
        accepted.size == 1 and accepted.dtype.kind == "b",
        f"{FN}: `accepted` must be a single bool, got dtype {accepted.dtype}",
    )
    return int(token.reshape(())), bool(accepted.reshape(()))


def decisive(p, q, x, u_accept, u_resample):
    """True when neither decision of this call is within MARGIN of its boundary."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    ratio = p[x] / q[x]
    # p[x] == 0 is exact in float32 too: such a token is rejected for every u_accept.
    if 0 < ratio < 1 + MARGIN and abs(u_accept - ratio) < MARGIN:
        return False
    cdf = np.cumsum(np.maximum(p - q, 0.0))
    if cdf[-1] <= 0:
        return True
    edges = cdf[cdf > 0] / cdf[-1]
    return bool(np.abs(edges - u_resample).min() >= MARGIN)


def check_call(ctx, p, q, x, u_accept, u_resample):
    if not decisive(p, q, x, u_accept, u_resample):  # an authoring error
        raise RuntimeError("ambiguous test input: a decision is within MARGIN of its boundary")
    got = step(ctx, p, q, x, u_accept, u_resample)
    want = ctx.expected(FN, p, q, int(x), float(u_accept), float(u_resample))
    detail = (
        f"call: {FN}(p, q, x, u_accept, u_resample)\n  p = {p}\n  q = {q}\n  x = {x}\n"
        f"  u_accept = {u_accept}\n  u_resample = {u_resample}\n"
        f"expected (token, accepted) = {(int(want[0]), bool(want[1]))}\ngot {got}"
    )
    check.that(got[1] == bool(want[1]), f"{FN}: wrong acceptance decision", detail)
    check.that(got[0] == int(want[0]), f"{FN}: wrong token", detail)


def distributions(ctx, vocab, concentration=1.0):
    """A random (p, q) pair as float32 vectors that sum to 1."""
    pair = ctx.rng.dirichlet(np.full(vocab, concentration), size=2).astype(np.float32)
    pair /= pair.sum(axis=1, keepdims=True)
    return pair[0], pair[1]


def run_steps(ctx, p, q, drafts):
    """One call per draft token, each with its own pair of fresh uniforms."""
    uniforms = ctx.rng.random((len(drafts), 2))
    tokens, accepted = [], []
    for x, (u_accept, u_resample) in zip(drafts, uniforms, strict=True):
        token, ok = step(ctx, p, q, x, u_accept, u_resample)
        check.that(
            ok == (token == x),
            f"{FN}: `accepted` disagrees with the token (an accepted token is the draft "
            "token; a resampled token never is)",
            f"x = {x}, returned (token, accepted) = {(token, ok)}",
        )
        tokens.append(token)
        accepted.append(ok)
    return tokens, accepted


def exact(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return probs / probs.sum()


@case("the worked example: one acceptance, two rejections, one certain acceptance", public=True)
def worked_example(ctx):
    calls = [
        ((0, 0.1, 0.9), (0, True)),  # 0.1 < p[0] / q[0] = 0.2: accepted
        ((0, 0.6, 0.3), (1, False)),  # rejected; 0.3 falls in token 1's half of the residual
        ((0, 0.6, 0.7), (3, False)),  # rejected; 0.7 falls in token 3's half
        ((1, 0.99, 0.5), (1, True)),  # p[1] >= q[1]: accepted whatever u_accept is
    ]
    for (x, u_accept, u_resample), want in calls:
        got = step(ctx, P4, Q4, x, u_accept, u_resample)
        check.that(
            got == want,
            f"{FN}: wrong result for x = {x}, u_accept = {u_accept}, u_resample = {u_resample}",
            f"p = {P4}\nq = {Q4}\nexpected (token, accepted) = {want}\ngot {got}",
        )


@case("which uniform decides what: a grid of u_accept and u_resample for every x", public=True)
def uniform_grid(ctx):
    # Catches a swapped or reused uniform, the inverted ratio, and a residual that is
    # not normalized, call by call.
    p, q = distributions(ctx, 6)
    grid = [0.03, 0.21, 0.42, 0.58, 0.77, 0.96]
    for x in range(6):
        for u_accept in grid:
            for u_resample in reversed(grid):
                if decisive(p, q, x, u_accept, u_resample):
                    check_call(ctx, p, q, x, u_accept, u_resample)


@case("losslessness: with x drawn from q, the emitted token is distributed as p", public=True)
def output_follows_p(ctx):
    p, q = distributions(ctx, 8)
    drafts = ctx.rng.choice(8, size=2000, p=exact(q))
    tokens, accepted = run_steps(ctx, p, q, drafts)
    check.distribution(tokens, exact(p), label="emitted token")
    # The acceptance rate is the overlap of the two distributions (Leviathan et al.).
    overlap = float(np.minimum(exact(p), exact(q)).sum())
    check.mean(np.array(accepted, dtype=np.float64), overlap, low=0, high=1, label="accepted")


@case("target equals draft: every token is accepted")
def identical_distributions(ctx):
    p, _ = distributions(ctx, 6)
    for x in range(6):
        for u_accept in (0.0, 0.5, 0.999):
            got = step(ctx, p, p.copy(), x, u_accept, 0.5)
            check.that(
                got == (x, True),
                f"{FN}: with p equal to q the draft token must always be accepted",
                f"x = {x}, u_accept = {u_accept}, got {got}",
            )


@case("a draft token the target gives probability 0 is rejected even when u_accept is 0")
def impossible_draft(ctx):
    p = np.array([0.0, 0.5, 0.0, 0.3, 0.2, 0.0], dtype=np.float32)
    q = np.array([0.4, 0.1, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
    for x in (0, 2, 5):
        for u_accept in (0.0, 0.4):
            check_call(ctx, p, q, x, u_accept, 0.37)


@case("resampling never returns a token whose residual is 0, for u_resample from 0 to 0.999")
def zero_residual_tokens(ctx):
    # Residual max(0, p - q) = [0, 0.3, 0, 0, 0.3, 0]: zeros before, between, and after
    # the two candidates. u_resample = 0 must give token 1 and 0.999 token 4.
    p = np.array([0.05, 0.4, 0.1, 0.0, 0.4, 0.05], dtype=np.float32)
    q = np.array([0.3, 0.1, 0.1, 0.3, 0.1, 0.1], dtype=np.float32)
    for u_resample in (0.0, 0.2, 0.45, 0.55, 0.8, 0.999):
        check_call(ctx, p, q, 3, 0.5, u_resample)
        check_call(ctx, p, q, 0, 0.9, u_resample)


@case("a draft far from the target, with tokens the target excludes: the output still follows p")
def poor_draft(ctx):
    # Only 17% of the drafts are accepted; the rest of p's mass comes from resampling.
    p = np.array([0.0, 0.02, 0.5, 0.3, 0.0, 0.1, 0.08, 0.0], dtype=np.float32)
    q = np.array([0.3, 0.4, 0.05, 0.05, 0.1, 0.02, 0.03, 0.05], dtype=np.float32)
    drafts = ctx.rng.choice(8, size=1500, p=exact(q))
    tokens, accepted = run_steps(ctx, p, q, drafts)
    check.distribution(tokens, exact(p), label="emitted token")
    overlap = float(np.minimum(exact(p), exact(q)).sum())
    check.mean(np.array(accepted, dtype=np.float64), overlap, low=0, high=1, label="accepted")


@case("after a rejection the token follows the normalized residual max(0, p - q)")
def resample_follows_residual(ctx):
    p, q = distributions(ctx, 8, concentration=0.7)
    x = int(np.argmin(p / q))  # the token the draft overrates most; rejected for u_accept >= 0.9
    if p[x] / q[x] > 0.8:
        raise RuntimeError("authoring error: the draft token is not reliably rejected")
    uniforms = ctx.rng.random((1000, 2))
    tokens = []
    for u_accept, u_resample in uniforms:
        token, ok = step(ctx, p, q, x, 0.9 + 0.1 * u_accept, u_resample)
        check.that(not ok, f"{FN}: accepted a draft token although u_accept >= p[x] / q[x]")
        tokens.append(token)
    residual = np.maximum(p.astype(np.float64) - q.astype(np.float64), 0.0)
    check.distribution(tokens, exact(residual), label="resampled token")
