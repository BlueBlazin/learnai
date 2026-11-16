"""Tests for token-mask-apply.

Numeric cases compare both functions with the oracle. The property case states what a
masked distribution is without reference to any implementation: disallowed tokens have
probability exactly 0, allowed logits pass through untouched, and the allowed
probabilities are the softmax over the allowed tokens alone (computed here in float64).
"""

import numpy as np

from learnai.judge import case, check

BOTH = ("mask_logits", "masked_probs")
NEG_INF = float("-inf")


def check_both(ctx, logits, allowed):
    logits = np.asarray(logits, dtype=np.float32)
    allowed = np.asarray(allowed, dtype=bool)
    for fn in BOTH:
        ctx.check(fn, logits, allowed)


def random_mask(rng, shape, keep=0.5):
    """A random boolean mask with at least one allowed token in every row."""
    allowed = rng.random(shape) < keep
    rows = np.arange(shape[0])
    allowed[rows, rng.integers(0, shape[1], size=shape[0])] = True
    return allowed


@case("the worked example: two rows, two different masks", public=True)
def worked_example(ctx):
    logits = np.array([[2.0, 1.0, 0.0, 3.0], [0.5, 0.5, 4.0, 0.5]], dtype=np.float32)
    allowed = np.array([[True, True, False, False], [True, False, False, True]])
    masked = ctx.call("mask_logits", logits, allowed)
    expected = np.array([[2.0, 1.0, NEG_INF, NEG_INF], [0.5, NEG_INF, NEG_INF, 0.5]], np.float32)
    check.allclose(masked, expected, label="mask_logits", dtype=True)
    probs = ctx.call("masked_probs", logits, allowed)
    e = float(np.exp(1.0))
    expected = np.array([[e / (1 + e), 1 / (1 + e), 0.0, 0.0], [0.5, 0.0, 0.0, 0.5]], np.float32)
    check.allclose(probs, expected, label="masked_probs", dtype=True)


@case("a random batch: every row has its own mask", public=True)
def random_batch(ctx):
    # Rejects shared_mask (one mask for the whole batch) and no_renormalize.
    logits = ctx.randn(5, 12, scale=2.0)
    check_both(ctx, logits, random_mask(ctx.rng, logits.shape))


@case("the model's favorite token is disallowed by a wide margin", public=True)
def favorite_disallowed(ctx):
    # Rejects mask_after_softmax: over all tokens, the allowed ones have probability
    # exp(-300), which is 0 in float32, so renormalizing afterward divides 0 by 0.
    logits = np.array(
        [[300.0, 0.0, 1.0, -2.0, 300.0], [-50.0, 250.0, -51.0, -52.0, 249.0]], dtype=np.float32
    )
    allowed = np.array([[False, True, True, True, False], [True, False, True, True, False]])
    check_both(ctx, logits, allowed)


@case("a row with no allowed token raises ValueError", public=True)
def empty_row_raises(ctx):
    # Rejects all_masked_nan.
    logits = ctx.randn(1, 6)
    allowed = np.zeros((1, 6), dtype=bool)
    for fn in BOTH:
        check.raises(ValueError, ctx.call, fn, logits, allowed, label=f"{fn} with an empty row")


@case("one dead row among live ones still raises ValueError")
def one_dead_row(ctx):
    # Rejects checks_whole_batch: the batch as a whole has allowed tokens.
    logits = ctx.randn(4, 7)
    for dead in (0, 2, 3):
        allowed = random_mask(ctx.rng, logits.shape)
        allowed[dead] = False
        for fn in BOTH:
            check.raises(ValueError, ctx.call, fn, logits, allowed, label=f"{fn} with a dead row")
    # A full mask and a mask with one token per row are fine.
    check_both(ctx, logits, np.ones(logits.shape, dtype=bool))


@case("masked entries are -inf, not a large negative number")
def exact_infinity(ctx):
    # Rejects large_negative_constant.
    logits = ctx.randn(3, 9, scale=5.0)
    allowed = random_mask(ctx.rng, logits.shape, keep=0.4)
    got = ctx.call("mask_logits", logits, allowed)
    check.shape(got, logits.shape, label="mask_logits")
    check.that(
        bool(np.all(np.isneginf(got[~allowed]))),
        "mask_logits: a disallowed entry is not -inf",
        f"allowed =\n{allowed}\ngot =\n{got}\n",
    )
    check.that(
        np.array_equal(got[allowed], logits[allowed]),
        "mask_logits: an allowed logit was changed",
        f"allowed =\n{allowed}\nlogits =\n{logits}\ngot =\n{got}\n",
    )


@case("logits of magnitude 1e4 on both sides of the mask")
def large_magnitudes(ctx):
    logits = ctx.uniform(4, 8, low=-1e4, high=1e4)
    check_both(ctx, logits, random_mask(ctx.rng, logits.shape))
    # Allowed logits near -1e4 under a disallowed logit of +1e4.
    logits = np.array([[1e4, -1e4, -9999.0, -9998.5], [-1e4, 1e4, 9999.5, 1e4]], dtype=np.float32)
    allowed = np.array([[False, True, True, True], [True, False, True, False]])
    check_both(ctx, logits, allowed)


@case("logits that already contain -inf from an earlier filter")
def already_masked(ctx):
    logits = ctx.randn(4, 10, scale=2.0)
    logits[ctx.rng.random(logits.shape) < 0.3] = NEG_INF
    allowed = random_mask(ctx.rng, logits.shape, keep=0.6)
    # Every row keeps one allowed token with a finite logit (a guarantee of the statement).
    cols = ctx.rng.integers(0, 10, size=4)
    logits[np.arange(4), cols] = ctx.randn(4)
    allowed[np.arange(4), cols] = True
    check_both(ctx, logits, allowed)


@case("a single allowed token per row, a full mask, a single row, a single column")
def degenerate_masks(ctx):
    logits = ctx.randn(6, 6, scale=3.0)
    one_hot = np.zeros((6, 6), dtype=bool)
    one_hot[np.arange(6), ctx.rng.permutation(6)] = True
    check_both(ctx, logits, one_hot)
    probs = ctx.call("masked_probs", logits, one_hot)
    check.equal(probs, one_hot.astype(np.float32), label="masked_probs with one allowed token")
    check_both(ctx, logits, np.ones((6, 6), dtype=bool))
    check_both(ctx, logits[:1], random_mask(ctx.rng, (1, 6)))
    check_both(ctx, logits[:, :1], np.ones((6, 1), dtype=bool))


@case("property: disallowed mass is exactly 0 and the rest is the softmax over allowed tokens")
def restricted_softmax(ctx):
    shapes = [(1, 5), (3, 8), (4, 16), (2, 33)]

    def prop(rng):
        shape = shapes[int(rng.integers(len(shapes)))]
        logits = ctx.randn(*shape, scale=float(10.0 ** rng.uniform(-1, 2.5)))
        allowed = random_mask(rng, shape, keep=float(rng.uniform(0.1, 0.9)))
        detail = f"logits =\n{logits}\nallowed =\n{allowed}\n"

        masked = ctx.call("mask_logits", logits, allowed)
        check.shape(masked, shape, label="mask_logits")
        check.that(
            bool(np.all(np.isneginf(masked[~allowed]))),
            "mask_logits: a disallowed entry is not -inf",
            detail,
        )
        check.that(
            np.array_equal(masked[allowed], logits[allowed]),
            "mask_logits: an allowed logit was changed",
            detail,
        )

        probs = ctx.call("masked_probs", logits, allowed)
        check.shape(probs, shape, label="masked_probs")
        check.finite(probs, label="masked_probs")
        check.that(
            bool(np.all(probs[~allowed] == 0.0)),
            "masked_probs: a disallowed token has non-zero probability",
            detail + f"got =\n{probs}\n",
        )
        sums = probs.sum(axis=-1, dtype=np.float64)
        check.allclose(sums, np.ones_like(sums), atol=1e-5, label="row sums of masked_probs")
        # The definition, in float64: softmax over the allowed entries only.
        x = np.where(allowed, logits.astype(np.float64), -np.inf)
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        check.allclose(probs, e / e.sum(axis=-1, keepdims=True), label="masked_probs")

    check.forall(prop, trials=12)
