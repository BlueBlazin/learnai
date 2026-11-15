"""Tests for multi-draft-rejection.

The numeric cases pin down how the uniforms are used, by comparing single calls with
`definition`, a float64 loop written straight from the statement. Their inputs stay
clear of every decision boundary (`decisive` checks it), so float32 rounding can never
decide a call.

The statistical cases check losslessness. The test plays the draft model: the n
candidates are drawn independently from q, and all uniforms are independent. For a
correct solution:

- the emitted token is distributed as p, whatever q and n are;
- the number of rejected candidates k has P(k >= j) = (1 - beta_0) ... (1 - beta_{j-1}),
  with beta_i = sum(min(r_i, q)) and r_i the residual after i rejections (a fixed
  sequence: with independent candidates it does not depend on which tokens were
  rejected);
- given that all n candidates were rejected, the token is distributed as r_n.

There are nine statistical checks in this file, each with the default delta of 1e-10,
which keeps the false-failure rate of a correct solution below 1e-9.
"""

import numpy as np

from whetstone.judge import case, check

FN = "multi_draft_step"
MARGIN = 1e-3

# The worked example of the statement.
P = np.array([0.1, 0.4, 0.2, 0.3], dtype=np.float32)
Q = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float32)


def exact(row):
    row = np.asarray(row, dtype=np.float64)
    return row / row.sum()


def residuals(p, q, n):
    """r_0 .. r_n in float64: r_0 = p, r_{i+1} = norm(max(0, r_i - q))."""
    rows = [exact(p)]
    q = exact(q)
    for _ in range(n):
        rest = np.maximum(rows[-1] - q, 0.0)
        if rest.sum() <= 1e-9:  # only when r_i equals q: no rejection can happen there
            rest = rows[-1]
        rows.append(rest / rest.sum())
    return rows


def definition(p, q, candidates, u_accept, u_resample):
    """((token, n_rejected), decisive) by the statement, in float64."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    decisive = True
    residual = p / p.sum()
    for i, x in enumerate(candidates):
        ratio = residual[x] / q[x]
        # A residual of 0 is exact in float32 too: rejected for every uniform.
        if residual[x] > 1e-7 and ratio < 1 + MARGIN and abs(u_accept[i] - ratio) < MARGIN:
            decisive = False
        if 0 < residual[x] <= 1e-7:  # a residual entry that should be 0 and is not quite
            decisive = False
        if u_accept[i] * q[x] < residual[x]:
            return (int(x), i), decisive
        residual = np.maximum(residual - q, 0.0)
        residual = residual / residual.sum()
    cdf = np.cumsum(residual)
    edges = cdf[cdf > 0] / cdf[-1]
    if (
        np.abs(edges - u_resample).min() < MARGIN
        or (0 < residual).any()
        and (residual[residual > 0].min() < 1e-6)
    ):
        decisive = False
    return (int(np.sum(cdf <= u_resample * cdf[-1])), len(candidates)), decisive


def call(ctx, p, q, candidates, u_accept, u_resample):
    """Call the solution and return `(token, n_rejected)` as Python ints."""
    result = ctx.call(
        FN,
        p,
        q,
        np.asarray(candidates, dtype=np.int64),
        np.asarray(u_accept, dtype=np.float32),
        float(u_resample),
    )
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{FN}: expected a (token, n_rejected) pair, got {type(result).__name__}",
    )
    values = [np.asarray(value) for value in result]
    check.that(
        all(value.size == 1 and value.dtype.kind in "iu" for value in values),
        f"{FN}: token and n_rejected must each be a single integer",
        f"got {result!r}",
    )
    token, rejected = (int(value.reshape(())) for value in values)
    check.that(0 <= token < p.shape[0], f"{FN}: token is not a token id", f"got {token}")
    check.that(
        0 <= rejected <= len(candidates), f"{FN}: n_rejected is outside 0..n", f"got {rejected}"
    )
    return token, rejected


def check_call(ctx, p, q, candidates, u_accept, u_resample, literal=None):
    want, decisive = definition(p, q, candidates, u_accept, u_resample)
    if not decisive or (literal is not None and literal != want):  # an authoring error
        raise RuntimeError("ambiguous test input, or a wrong literal in the statement")
    oracle = ctx.expected(
        FN, p, q, np.asarray(candidates), np.asarray(u_accept, dtype=np.float32), float(u_resample)
    )
    if tuple(int(v) for v in oracle) != want:  # an authoring error
        raise RuntimeError("the reference disagrees with the definition")
    got = call(ctx, p, q, candidates, u_accept, u_resample)
    detail = (
        f"call: {FN}(p, q, candidates, u_accept, u_resample)\n  p = {p}\n  q = {q}\n"
        f"  candidates = {list(map(int, candidates))}\n"
        f"  u_accept = {[round(float(u), 4) for u in u_accept]}\n"
        f"  u_resample = {u_resample}\n"
        f"expected (token, n_rejected) = {want}\ngot {got}"
    )
    check.that(got[1] == want[1], f"{FN}: wrong number of rejected candidates", detail)
    check.that(got[0] == want[0], f"{FN}: wrong token", detail)


def models(ctx, vocab, closeness, alpha=1.5):
    """A random target p, and a draft q that blends p with another draw."""
    p = ctx.rng.dirichlet(np.full(vocab, alpha))
    q = closeness * p + (1 - closeness) * ctx.rng.dirichlet(np.full(vocab, alpha))
    return exact(p).astype(np.float32), exact(q).astype(np.float32)


def check_lossless(ctx, p, q, n, n_calls, *, tail=False):
    """Draw n candidates from q per call; compare what is emitted with p."""
    vocab = p.shape[0]
    pe, qe = exact(p), exact(q)
    candidates = ctx.rng.choice(vocab, size=(n_calls, n), p=qe)
    u_accept = ctx.rng.random((n_calls, n))
    u_resample = ctx.rng.random(n_calls)
    tokens, rejected, fallback = [], [], []
    for xs, ua, ur in zip(candidates, u_accept, u_resample, strict=True):
        token, k = call(ctx, p, q, xs, ua, ur)
        check.that(
            k == n or token == xs[k],
            f"{FN}: the token is not the candidate that n_rejected says was accepted",
            f"candidates = {xs.tolist()}, returned (token, n_rejected) = {(token, k)}",
        )
        check.that(
            token not in xs[:k],
            f"{FN}: emitted a token that was rejected earlier in the same call",
            f"candidates = {xs.tolist()}, returned (token, n_rejected) = {(token, k)}",
        )
        tokens.append(token)
        rejected.append(k)
        if k == n:
            fallback.append(token)
    rows = residuals(pe, qe, n)
    beta = np.array([np.minimum(rows[i], qe).sum() for i in range(n)])
    reach = np.concatenate([[1.0], np.cumprod(1 - beta)])  # P(n_rejected >= j)
    check.distribution(tokens, pe, label="emitted token")
    check.distribution(
        rejected, reach * np.concatenate([beta, [1.0]]), label="number of rejected candidates"
    )
    if tail:
        check.distribution(fallback, rows[n], label=f"token after all {n} candidates were rejected")
    return len(fallback)


@case("the worked example: first candidate, second candidate, and the final residual",
      public=True)  # fmt: skip
def worked_example(ctx):
    check_call(ctx, P, Q, [0, 1], [0.1, 0.9], 0.5, literal=(0, 0))
    check_call(ctx, P, Q, [0, 1], [0.6, 0.9], 0.5, literal=(1, 1))
    check_call(ctx, P, Q, [0, 0], [0.6, 0.0], 0.5, literal=(3, 2))
    check_call(ctx, P, Q, [0, 2], [0.6, 0.3], 0.3, literal=(1, 2))
    check_call(ctx, P, Q, [2, 0], [0.99, 0.1], 0.5, literal=(2, 0))


@case("which uniform decides what: a grid of uniforms over three candidates", public=True)
def uniform_grid(ctx):
    # Catches a residual that is not updated or not normalized, a reused uniform, and
    # a candidate accepted after an earlier one already was, call by call.
    p, q = models(ctx, 5, closeness=0.2)
    grid = [0.03, 0.35, 0.65, 0.97]
    checked = 0
    for xs in ([0, 1, 2], [3, 4, 0], [2, 2, 1], [4, 3, 3], [1, 0, 4]):
        for u0 in grid:
            for u1 in reversed(grid):
                for u2, u_resample in ((0.2, 0.13), (0.8, 0.71)):
                    args = (p, q, xs, [u0, u1, u2], u_resample)
                    if definition(*args)[1]:
                        check_call(ctx, *args)
                        checked += 1
    if checked < 100:  # an authoring error
        raise RuntimeError("too few decisive grid points")


@case("losslessness: three candidates drawn from q, the emitted token is distributed as p",
      public=True)  # fmt: skip
def lossless_three(ctx):
    p, q = models(ctx, 6, closeness=0.3)
    check_lossless(ctx, p, q, 3, 700)


@case("one candidate: the rule is the single speculative sampling step")
def single_candidate(ctx):
    p, q = models(ctx, 6, closeness=0.4)
    for x in range(6):
        for u in (0.02, 0.5, 0.98):
            args = (p, q, [x], [u], 0.37)
            if definition(*args)[1]:
                check_call(ctx, *args)
    check_lossless(ctx, p, q, 1, 500)


@case("a token rejected once is never accepted later in the call, even with a uniform of 0")
def duplicate_candidates(ctx):
    # After token x is rejected its residual is exactly 0. A duplicate still uses up
    # a stage: the residual is updated again. Rejects subtracts_once_per_token and
    # residual_not_updated.
    p = np.array([0.05, 0.15, 0.3, 0.5], dtype=np.float32)
    q = np.array([0.45, 0.35, 0.1, 0.1], dtype=np.float32)
    check_call(ctx, p, q, [0, 0, 0], [0.5, 0.0, 0.0], 0.2)
    check_call(ctx, p, q, [0, 0, 0], [0.5, 0.0, 0.0], 0.8)
    check_call(ctx, p, q, [1, 1, 3], [0.9, 0.0, 0.99], 0.4)
    check_call(ctx, p, q, [0, 1, 0, 1], [0.6, 0.6, 0.0, 0.0], 0.55)


@case("target equals draft: the first candidate is accepted for every uniform")
def identical_models(ctx):
    p, _ = models(ctx, 6, closeness=1.0)
    for u in (0.0, 0.4, 0.999):
        xs = ctx.randint(0, 6, 4)
        check_call(ctx, p, p.copy(), xs, [u, 0.5, 0.5, 0.5], 0.3)


@case("candidates the target excludes are rejected even when their uniform is 0")
def impossible_candidates(ctx):
    p = np.array([0.5, 0.0, 0.3, 0.2, 0.0], dtype=np.float32)
    q = np.array([0.3, 0.3, 0.1, 0.1, 0.2], dtype=np.float32)
    check_call(ctx, p, q, [1, 4], [0.0, 0.0], 0.37)
    check_call(ctx, p, q, [1, 4], [0.0, 0.0], 0.93)
    check_call(ctx, p, q, [4, 0], [0.0, 0.9], 0.5)
    check_call(ctx, p, q, [1, 2, 4], [0.0, 0.99, 0.0], 0.5)


@case("every candidate rejected: the token comes from the last residual (n = 1 to 5)")
def all_rejected(ctx):
    # Rejects final_sample_from_p, residual_not_updated, and reuses_uniform call by
    # call. Token 0 is heavily overrated by the draft and always proposed.
    p = np.array([0.02, 0.08, 0.2, 0.3, 0.4], dtype=np.float32)
    q = np.array([0.6, 0.2, 0.1, 0.06, 0.04], dtype=np.float32)
    for n in range(1, 6):
        for u_resample in (0.05, 0.31, 0.58, 0.9):
            args = (p, q, [0] * n, [0.9] + [0.35] * (n - 1), u_resample)
            if definition(*args)[1]:
                check_call(ctx, *args)


@case("a poor draft and five candidates: the output still follows p, and the fallback "
      "follows the last residual")  # fmt: skip
def lossless_poor_draft(ctx):
    # The target excludes two tokens the draft likes; most calls reject several
    # candidates. A token with target probability 0 must never appear.
    p = np.array([0.0, 0.04, 0.36, 0.3, 0.0, 0.1, 0.12, 0.08], dtype=np.float32)
    q = np.array([0.25, 0.3, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05], dtype=np.float32)
    n_fallback = check_lossless(ctx, p, q, 5, 700, tail=True)
    if n_fallback < 150:  # an authoring error
        raise RuntimeError("too few calls reject every candidate")


@case("eight candidates over a 64-token vocabulary")
def many_candidates(ctx):
    p, q = models(ctx, 64, closeness=0.1, alpha=0.5)
    done = 0
    while done < 12:
        args = (p, q, ctx.randint(0, 64, 8), ctx.rng.random(8), float(ctx.rng.random()))
        if definition(*args)[1]:
            check_call(ctx, *args)
            done += 1


@case("two candidates and a draft close to the target: acceptance must not be undercounted")
def lossless_two(ctx):
    # The draft ranks the tokens in the opposite order: the first candidate is usually
    # rejected and the second stage carries most of the weight. Rejects
    # last_accepted_wins and independent_tests statistically.
    p = np.array([0.05, 0.1, 0.15, 0.2, 0.5], dtype=np.float32)
    q = np.array([0.5, 0.2, 0.15, 0.1, 0.05], dtype=np.float32)
    check_lossless(ctx, p, q, 2, 700)
