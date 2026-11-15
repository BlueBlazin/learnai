"""Tests for kl-divergence-from-logits.

Numeric cases compare with the oracle (and with literal values for the statement's
examples); property cases state what a divergence is without it. Every result is also
checked for the things the statement rules out everywhere: NaN and negative values.
"""

import numpy as np

from whetstone.judge import case, check

NINF = -np.inf
ATOL = 1e-5


def f32(values):
    return np.array(values, dtype=np.float32)


def kl(ctx, p, q, want=None):
    """Compare with the oracle (and with `want`); no NaN, nothing below zero."""
    got = np.asarray(ctx.check("kl_from_logits", p, q, atol=ATOL))
    check.that(not np.isnan(got).any(), "kl_from_logits: got NaN", f"got = {got!r}")
    check.that(
        bool((got >= 0).all()),
        "kl_from_logits: got a negative value; a divergence is never below 0",
        f"smallest value: {got.min()!r}",
    )
    if want is not None:
        check.allclose(got, np.asarray(want, dtype=np.float32), atol=ATOL, label="kl_from_logits")
    return got


def masked(ctx, x, fraction):
    """Mask a random subset of each row with -inf, always leaving one entry finite."""
    x = x.copy()
    mask = ctx.rng.random(x.shape) < fraction
    keep = ctx.rng.integers(0, x.shape[-1], size=x.shape[:-1])
    np.put_along_axis(mask, keep[..., None], False, axis=-1)
    x[mask] = NINF
    return x


# Rejects swapped_arguments: the two directions differ in the third decimal.
@case("the worked example: P = [0.5, 0.25, 0.25] against the uniform distribution", public=True)
def worked_example(ctx):
    p, q = f32([np.log(2.0), 0.0, 0.0]), f32([0.0, 0.0, 0.0])
    kl(ctx, p, q, 0.058892)
    kl(ctx, q, p, 0.056633)
    kl(ctx, p, p, 0.0)


# Rejects reduces_over_batch.
@case("a random batch, shape (4, 8): one value per row", public=True)
def random_batch(ctx):
    kl(ctx, ctx.randn(4, 8, scale=2.0), ctx.randn(4, 8, scale=2.0))


# Rejects nan_on_masked (rows 0 and 2) and skips_q_zero (row 1).
@case("masked tokens: 0 where p is masked, inf where only q is", public=True)
def masked_tokens(ctx):
    p = f32([[0.0, 0.0, NINF], [0.0, 0.0, NINF], [0.0, NINF, NINF]])
    q = f32([[0.0, 0.0, 0.0], [0.0, NINF, NINF], [0.0, 0.0, NINF]])
    kl(ctx, p, q, [np.log(1.5), np.inf, np.log(2.0)])


# Rejects from_probabilities.
@case("a probability that underflows float32 still has a finite divergence", public=True)
def underflowing_q(ctx):
    kl(ctx, f32([0.0, 0.0]), f32([0.0, -200.0]), 99.306854)
    p = f32([[1.0, 0.0, -1.0, 0.5], [3.0, 2.5, -150.0, 0.0]])
    q = f32([[0.0, -300.0, 2.0, -120.0], [-400.0, 1.0, 0.0, -90.0]])
    got = kl(ctx, p, q)
    check.finite(got, label="kl_from_logits")


@case("batch and sequence axes: shape (2, 5, 16) gives shape (2, 5)", public=True)
def batch_and_sequence(ctx):
    got = kl(ctx, ctx.randn(2, 5, 16, scale=3.0), ctx.randn(2, 5, 16, scale=3.0))
    check.shape(got, (2, 5), label="kl_from_logits")


# Rejects unshifted_logsumexp: with all logits of a row near 1e4, `x - logsumexp(x)`
# is off by up to 1e-3 per row, and by a different amount for p and for q.
@case("large logits: rows offset by 5e3 to 1e4, and rows spanning -1e4 to 1e4", public=True)
def large_logits(ctx):
    p = ctx.randn(8, 8, scale=1.5) + ctx.uniform(8, 1, low=5e3, high=1e4)
    q = ctx.randn(8, 8, scale=1.5) + ctx.uniform(8, 1, low=5e3, high=1e4)
    kl(ctx, p, q)
    kl(ctx, -p, q)
    p = f32([[1e4, 0.0, -1e4], [1e4, 9999.0, 0.0], [-1e4, -1e4, 1e4], [0.0, 1.0, 2.0]])
    q = f32([[-1e4, 0.0, 1e4], [9999.0, 1e4, 0.0], [1e4, -1e4, -1e4], [1e4, 0.0, -1e4]])
    got = kl(ctx, p, q)
    check.finite(got, label="kl_from_logits")


# Rejects underflow_counts_as_zero: exp(log p) is 0.0, but the logit is finite.
@case("q masks a token to which p gives a probability too small for float32")
def underflowing_p_masked_in_q(ctx):
    kl(ctx, f32([0.0, -200.0]), f32([0.0, NINF]), np.inf)
    p = f32([[5.0, -500.0, 0.0, NINF], [0.0, -1e4, 1.0, 2.0], [2.0, -300.0, NINF, 0.0]])
    q = f32([[1.0, NINF, 0.0, 2.0], [0.5, NINF, NINF, 0.0], [0.0, 1.0, NINF, -2.0]])
    kl(ctx, p, q)


# Rejects no_clamp: about half of these rows sum to a tiny negative number in float32.
@case("nearly identical distributions never give a negative divergence")
def nearly_identical(ctx):
    p = ctx.randn(64, 32, scale=3.0)
    q = p + ctx.randn(64, 32, scale=1e-4)
    kl(ctx, p, q)
    kl(ctx, q, p)
    same = kl(ctx, p, p.copy())
    check.that(bool((same == 0).all()), "kl_from_logits(p, p): got a nonzero value")


@case("a constant added to every logit of a row changes nothing")
def shifted_logits(ctx):
    p, q = ctx.randn(4, 8, scale=2.0), ctx.randn(4, 8, scale=2.0)
    base = kl(ctx, p, q)
    shifted = kl(ctx, p + f32([[100.0], [-50.0], [7.0], [0.0]]), q - np.float32(250.0))
    check.allclose(shifted, base, rtol=1e-3, atol=1e-4, label="kl_from_logits of shifted logits")
    zero = kl(ctx, p, p + f32([[3.0], [-3.0], [40.0], [0.5]]))
    check.allclose(zero, np.zeros(4, np.float32), atol=1e-4, label="kl_from_logits(p, p + c)")


@case("random masks in p, in q, and in both")
def random_masks(ctx):
    p, q = ctx.randn(2, 5, 16, scale=2.0), ctx.randn(2, 5, 16, scale=2.0)
    kl(ctx, masked(ctx, p, 0.4), q)  # finite: masks in p only
    kl(ctx, p, masked(ctx, q, 0.1))  # infinite wherever q masks anything
    mp = masked(ctx, ctx.randn(4, 8, scale=2.0), 0.5)
    mq = np.where(np.isneginf(mp) & (ctx.rng.random((4, 8)) < 0.7), NINF, ctx.randn(4, 8))
    kl(ctx, mp, mq.astype(np.float32))  # finite: q masks only what p masks
    kl(ctx, masked(ctx, ctx.randn(4, 8), 0.3), masked(ctx, ctx.randn(4, 8), 0.3))


@case("a single 1-D pair gives a 0-dimensional result; a vocabulary of one token gives 0")
def small_shapes(ctx):
    got = kl(ctx, ctx.randn(8, scale=2.0), ctx.randn(8, scale=2.0))
    check.shape(got, (), label="kl_from_logits")
    one = kl(ctx, ctx.randn(4, 1, scale=50.0), ctx.randn(4, 1, scale=50.0))
    check.that(bool((one == 0).all()), "kl_from_logits with V = 1: got a nonzero value")
    kl(ctx, ctx.randn(3, 4096), ctx.randn(3, 4096))


# What a divergence is, without the oracle: Gibbs' inequality, and the exact value
# against a point mass, KL(onehot_k || Q) = -log Q_k.
@case("KL is non-negative, and KL(one-hot at k || Q) equals -log Q[k]")
def gibbs_and_point_mass(ctx):
    def prop(rng):
        scale = float(10.0 ** rng.uniform(-3, 1.5))
        p, q = ctx.randn(4, 8, scale=scale), ctx.randn(4, 8, scale=scale)
        got = np.asarray(ctx.call("kl_from_logits", masked(ctx, p, 0.3), q))
        check.shape(got, (4,), label="kl_from_logits")
        check.finite(got, label="kl_from_logits")
        check.that(bool((got >= 0).all()), "kl_from_logits: got a negative value")
        k = rng.integers(0, 8, size=4)
        onehot = np.full((4, 8), NINF, dtype=np.float32)
        np.put_along_axis(onehot, k[:, None], rng.normal(size=(4, 1)).astype(np.float32), axis=-1)
        q64 = q.astype(np.float64)
        log_q = q64 - np.log(np.exp(q64 - q64.max(-1, keepdims=True)).sum(-1, keepdims=True))
        log_q -= q64.max(-1, keepdims=True)
        want = -np.take_along_axis(log_q, k[:, None], axis=-1)[:, 0]
        point = ctx.call("kl_from_logits", onehot, q)
        check.allclose(point, want, atol=ATOL, label="kl_from_logits(one-hot, q)")

    check.forall(prop, trials=10)


@case("random scales, shapes, and masks")
def random_sweep(ctx):
    for shape in ((4, 8), (2, 5, 16), (8,), (64, 32)):
        scale = float(10.0 ** ctx.rng.uniform(-2, 2.5))
        p, q = ctx.randn(*shape, scale=scale), ctx.randn(*shape, scale=scale)
        kl(ctx, p, q)
        kl(ctx, masked(ctx, p, 0.25), q)
