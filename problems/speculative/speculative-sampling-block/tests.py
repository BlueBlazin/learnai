"""Tests for speculative-sampling-block.

The numeric cases pin down how the uniforms are used, by comparing single calls with
`definition`, a float64 loop written straight from the statement. Their inputs stay
clear of every decision boundary (`decisive` checks it), so float32 rounding can never
decide a call.

The statistical cases check losslessness. The test plays both models: draft token i is
drawn from q[i], and all uniforms are independent. So that the calls are identically
distributed, the rows of p and q are fixed, as for models whose prediction depends on
the position only. Then, for a correct solution:

- the number of accepted drafts n has P(n >= j) = beta_0 * ... * beta_{j-1}, with
  beta_i = sum(min(p[i], q[i]));
- the token emitted at position j (draft j if it was accepted, else `next_token`), over
  the calls with n >= j, is distributed as p[j]. Position 0 is the first emitted token;
  position K is the bonus token. The event n >= j depends only on positions before j,
  so those calls are an independent sample for position j.

There are nine statistical checks in this file, each with the default delta of 1e-10,
which keeps the false-failure rate of a correct solution below 1e-9.
"""

import numpy as np

from learnai.judge import case, check

FN = "verify_block"
MARGIN = 1e-3

# The worked example of the statement.
P = np.array(
    [[0.10, 0.40, 0.20, 0.30], [0.25, 0.25, 0.25, 0.25], [0.50, 0.10, 0.10, 0.30]],
    dtype=np.float32,
)
Q = np.array([[0.50, 0.20, 0.20, 0.10], [0.10, 0.60, 0.20, 0.10]], dtype=np.float32)


def definition(p, q, drafts, u_accept, u_resample):
    """((n_accepted, next_token), decisive) by the statement, in float64."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    decisive = True
    n = 0
    while n < len(drafts):
        ratio = p[n, drafts[n]] / q[n, drafts[n]]
        # p == 0 is exact in float32 too: such a draft is rejected for every uniform.
        if 0 < ratio < 1 + MARGIN and abs(u_accept[n] - ratio) < MARGIN:
            decisive = False
        if not u_accept[n] * q[n, drafts[n]] < p[n, drafts[n]]:
            break
        n += 1
    weights = np.maximum(p[n] - q[n], 0.0) if n < len(drafts) else p[n]
    cdf = np.cumsum(weights)
    edges = cdf[cdf > 0] / cdf[-1]
    if np.abs(edges - u_resample).min() < MARGIN:
        decisive = False
    return (n, int(np.sum(cdf <= u_resample * cdf[-1]))), decisive


def call(ctx, p, q, drafts, u_accept, u_resample):
    """Call the solution and return `(n_accepted, next_token)` as Python ints."""
    result = ctx.call(
        FN,
        p,
        q,
        np.asarray(drafts, dtype=np.int64),
        np.asarray(u_accept, dtype=np.float32),
        float(u_resample),
    )
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{FN}: expected an (n_accepted, next_token) pair, got {type(result).__name__}",
    )
    values = [np.asarray(value) for value in result]
    check.that(
        all(value.size == 1 and value.dtype.kind in "iu" for value in values),
        f"{FN}: n_accepted and next_token must each be a single integer",
        f"got {result!r}",
    )
    n, token = (int(value.reshape(())) for value in values)
    check.that(0 <= n <= len(drafts), f"{FN}: n_accepted is outside 0..K", f"got {n}")
    check.that(0 <= token < p.shape[1], f"{FN}: next_token is not a token id", f"got {token}")
    return n, token


def check_call(ctx, p, q, drafts, u_accept, u_resample, literal=None):
    want, decisive = definition(p, q, drafts, u_accept, u_resample)
    if not decisive or (literal is not None and literal != want):  # an authoring error
        raise RuntimeError("ambiguous test input, or a wrong literal in the statement")
    oracle = ctx.expected(
        FN, p, q, np.asarray(drafts), np.asarray(u_accept, dtype=np.float32), float(u_resample)
    )
    if tuple(int(v) for v in oracle) != want:  # an authoring error
        raise RuntimeError("the reference disagrees with the definition")
    got = call(ctx, p, q, drafts, u_accept, u_resample)
    detail = (
        f"call: {FN}(p, q, draft_tokens, u_accept, u_resample)\n  p =\n{p}\n  q =\n{q}\n"
        f"  draft_tokens = {list(map(int, drafts))}\n"
        f"  u_accept = {[round(float(u), 4) for u in u_accept]}\n"
        f"  u_resample = {u_resample}\n"
        f"expected (n_accepted, next_token) = {want}\ngot {got}"
    )
    check.that(got[0] == want[0], f"{FN}: wrong number of accepted drafts", detail)
    check.that(got[1] == want[1], f"{FN}: wrong next token", detail)


def exact(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return rows / rows.sum(axis=-1, keepdims=True)


def models(ctx, k, vocab, closeness):
    """Random rows: p of shape (k + 1, vocab), and q a blend of p and another draw."""
    p = ctx.rng.dirichlet(np.full(vocab, 1.5), size=k + 1)
    q = closeness * p[:k] + (1 - closeness) * ctx.rng.dirichlet(np.full(vocab, 1.5), size=k)
    return exact(p).astype(np.float32), exact(q).astype(np.float32)


def check_lossless(ctx, p, q, n_calls):
    """Sample drafts from q, verify them, and compare what is emitted with p."""
    k, vocab = q.shape
    pe, qe = exact(p), exact(q)
    drafts = np.stack([ctx.rng.choice(vocab, size=n_calls, p=qe[i]) for i in range(k)], axis=1)
    u_accept = ctx.rng.random((n_calls, k))
    u_resample = ctx.rng.random(n_calls)
    emitted = [[] for _ in range(k + 1)]
    accepted = []
    for x, ua, ur in zip(drafts, u_accept, u_resample, strict=True):
        n, token = call(ctx, p, q, x, ua, ur)
        check.that(
            n == k or token != x[n],
            f"{FN}: after a rejection the next token can never be the rejected draft token",
            f"draft_tokens = {x.tolist()}, returned (n_accepted, next_token) = {(n, token)}",
        )
        accepted.append(n)
        for j in range(n):
            emitted[j].append(int(x[j]))
        emitted[n].append(token)
    beta = np.minimum(pe[:k], qe).sum(axis=1)
    reach = np.concatenate([[1.0], np.cumprod(beta)])  # P(n >= j)
    n_probs = reach * np.concatenate([1 - beta, [1.0]])
    check.distribution(accepted, n_probs, label="n_accepted")
    check.distribution(emitted[0], pe[0], label="first emitted token")
    for j in range(1, k):
        check.distribution(
            emitted[j], pe[j], label=f"token after an accepted prefix of {j} drafts"
        )
    check.distribution(emitted[k], pe[k], label=f"bonus token after all {k} drafts")
    return [len(tokens) for tokens in emitted]


@case("the worked example: full acceptance, a late and an early rejection, a bonus", public=True)
def worked_example(ctx):
    check_call(ctx, P, Q, [1, 1], [0.9, 0.3], 0.65, literal=(2, 2))
    check_call(ctx, P, Q, [1, 1], [0.9, 0.7], 0.50, literal=(1, 2))
    check_call(ctx, P, Q, [0, 1], [0.6, 0.1], 0.70, literal=(0, 3))
    check_call(ctx, P, Q, [0, 3], [0.1, 0.99], 0.20, literal=(2, 0))


@case("which uniform decides what: a grid of uniforms over two-token blocks", public=True)
def uniform_grid(ctx):
    # Catches a reused or swapped uniform, a draft accepted after a rejection, and a
    # correction or bonus sampled from the wrong weights, call by call.
    p, q = models(ctx, 2, 5, closeness=0.3)
    grid = [0.04, 0.5, 0.96]
    checked = 0
    for x0 in (0, 2, 4):
        for x1 in (1, 3):
            for u0 in grid:
                for u1 in reversed(grid):
                    for u_resample in (0.13, 0.71):
                        args = (p, q, [x0, x1], [u0, u1], u_resample)
                        if definition(*args)[1]:
                            check_call(ctx, *args)
                            checked += 1
    if checked < 60:  # an authoring error
        raise RuntimeError("too few decisive grid points")


@case("losslessness: drafts sampled from q, emitted tokens distributed as p (K = 3)",
      public=True)
def lossless_block(ctx):
    p, q = models(ctx, 3, 6, closeness=0.6)
    sizes = check_lossless(ctx, p, q, 600)
    if sizes[3] < 150:  # an authoring error
        raise RuntimeError("too few fully accepted blocks to test the bonus token")


@case("drafts after the first rejection are discarded, although each would pass its own test")
def later_drafts_discarded(ctx):
    # Rejects continues_after_rejection. Token 0 is overrated by the draft at every
    # position (p/q = 0.25), token 1 is underrated (always accepted).
    p = np.tile(np.array([0.1, 0.5, 0.25, 0.15], dtype=np.float32), (5, 1))
    q = np.tile(np.array([0.4, 0.2, 0.25, 0.15], dtype=np.float32), (4, 1))
    for position in range(4):
        drafts = [1, 1, 1, 1]
        drafts[position] = 0
        u_accept = [0.05, 0.05, 0.05, 0.05]
        u_accept[position] = 0.6
        check_call(ctx, p, q, drafts, u_accept, 0.4)
    check_call(ctx, p, q, [0, 0, 0, 0], [0.1, 0.2, 0.9, 0.1], 0.8)
    check_call(ctx, p, q, [0, 0, 0, 0], [0.1, 0.2, 0.1, 0.2], 0.8)


@case("target equals draft: everything is accepted and the bonus is sampled from p[K]")
def identical_models(ctx):
    # Rejects bonus_from_residual (its residual is empty here) and
    # bonus_from_last_draft_row (the last two rows of p differ).
    p, _ = models(ctx, 3, 6, closeness=1.0)
    for u_resample in (0.0, 0.11, 0.35, 0.62, 0.88, 0.999):
        drafts = ctx.randint(0, 6, 3)
        args = (p, p[:3].copy(), drafts, ctx.rng.random(3), u_resample)
        if definition(*args)[1]:
            check_call(ctx, *args)


@case("a bonus distribution with zeros, for u_resample from 0 to 0.999")
def bonus_with_zeros(ctx):
    # p[1] = [0, 0.3, 0, 0, 0.7, 0]: u_resample = 0 must give token 1, 0.999 token 4.
    p = np.array(
        [[0.2, 0.2, 0.2, 0.2, 0.1, 0.1], [0.0, 0.3, 0.0, 0.0, 0.7, 0.0]], dtype=np.float32
    )
    q = np.array([[0.1, 0.1, 0.3, 0.3, 0.1, 0.1]], dtype=np.float32)
    for u_resample in (0.0, 0.15, 0.45, 0.8, 0.999):
        check_call(ctx, p, q, [0], [0.97], u_resample)
        check_call(ctx, p, q, [4], [0.5], u_resample)


@case("a draft token the target excludes is rejected even when its uniform is 0")
def impossible_draft(ctx):
    p = np.array(
        [[0.5, 0.0, 0.3, 0.2], [0.0, 0.6, 0.0, 0.4], [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float32,
    )
    q = np.array([[0.4, 0.3, 0.2, 0.1], [0.3, 0.2, 0.3, 0.2]], dtype=np.float32)
    check_call(ctx, p, q, [1, 1], [0.0, 0.0], 0.37)
    check_call(ctx, p, q, [0, 2], [0.0, 0.0], 0.37)
    check_call(ctx, p, q, [0, 0], [0.5, 0.0], 0.77)
    check_call(ctx, p, q, [2, 3], [0.9, 0.9], 0.77)


@case("K = 1 and K = 8 with a 64-token vocabulary")
def block_sizes(ctx):
    for k in (1, 8):
        p, q = models(ctx, k, 64, closeness=0.7)
        done = 0
        while done < 5:
            args = (p, q, ctx.randint(0, 64, k), ctx.rng.random(k), float(ctx.rng.random()))
            if definition(*args)[1]:
                check_call(ctx, *args)
                done += 1


@case("a poor draft that proposes tokens the target excludes: the output still follows p")
def lossless_poor_draft(ctx):
    # Most blocks are cut short; whatever is emitted must still be distributed as p,
    # and a token with target probability 0 must never appear.
    p = np.array(
        [
            [0.0, 0.05, 0.45, 0.3, 0.0, 0.1, 0.1, 0.0],
            [0.3, 0.0, 0.0, 0.2, 0.25, 0.0, 0.05, 0.2],
            [0.0, 0.4, 0.1, 0.0, 0.0, 0.35, 0.0, 0.15],
        ],
        dtype=np.float32,
    )
    q = np.array(
        [
            [0.15, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05],
            [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    sizes = check_lossless(ctx, p, q, 450)
    if sizes[2] < 60:  # an authoring error
        raise RuntimeError("too few fully accepted blocks to test the bonus token")
