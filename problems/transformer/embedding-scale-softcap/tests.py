"""Tests for embedding-scale-softcap. Numeric cases compare the solution with the oracle.

The cases use three shapes of hidden states throughout, (2, 5, 8), (5, 8), and (8,),
with a vocabulary of 12, so that JAX compiles each operation a few times only.
"""

import numpy as np

from whetstone.judge import case, check

TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
V, D = 12, 8


def table(ctx, scale=0.5):
    return ctx.randn(V, D, scale=scale)


@case("the worked example: two lookups, and three logits divided by 4 and capped at 2", public=True)
def worked_example(ctx):
    emb = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]], dtype=np.float32)
    got = ctx.check("embed", np.array([2, 0]), emb, 12.0, **TOL)
    check.allclose(got, np.array([[6.0, -12.0], [12.0, 24.0]], dtype=np.float32), label="embed")
    h = np.array([3.0, 1.0], dtype=np.float32)
    got = ctx.check("finalize_logits", h, emb, 4.0, **TOL)
    check.allclose(got, np.array([1.25, 3.25, 0.125], dtype=np.float32), label="finalize_logits")
    got = ctx.check("finalize_logits", h, emb, 4.0, 2.0, **TOL)
    expected = np.array([1.109199, 1.850692, 0.124837], dtype=np.float32)
    check.allclose(got, expected, atol=1e-5, label="finalize_logits with softcap = 2")


# Rejects hardcodes_sqrt_d and logit_scale_multiplies.
@case("Granite's constants: embeddings times 12, logits divided by 4, no cap", public=True)
def granite_constants(ctx):
    emb = table(ctx)
    ctx.check("embed", ctx.randint(0, V, (2, 5)), emb, 12.0, **TOL)
    ctx.check("finalize_logits", ctx.randn(2, 5, D), emb, 4.0, **TOL)
    ctx.check("finalize_logits", ctx.randn(2, 5, D), emb, 4.0, None, **TOL)
    ctx.check("finalize_logits", ctx.randn(2, 5, D), emb, logit_scale=4.0, softcap=None, **TOL)


# Rejects hard_clip and tanh_without_inner_division: logits from far inside the cap
# to far outside it.
@case("Gemma's constants: embeddings times sqrt(d), logits capped at 30", public=True)
def gemma_constants(ctx):
    emb = table(ctx)
    ctx.check("embed", ctx.randint(0, V, (2, 5)), emb, float(np.sqrt(D)), **TOL)
    h = ctx.randn(2, 5, D, scale=20.0)  # logits of order 30
    got = ctx.check("finalize_logits", h, emb, 1.0, 30.0, **TOL)
    check.that(
        bool(np.all(np.abs(got) <= 30.0)),
        "finalize_logits: a capped logit is larger in magnitude than the cap",
    )
    ctx.check("finalize_logits", ctx.randn(2, 5, D, scale=2.0), emb, 1.0, 30.0, **TOL)


# Rejects cap_before_divisor: the raw logits are of order 40, the divided ones of
# order 10, and the cap of 8 sits between.
@case("both a divisor and a cap: the division comes first", public=True)
def division_then_cap(ctx):
    emb = table(ctx)
    h = ctx.randn(2, 5, D, scale=30.0)
    ctx.check("finalize_logits", h, emb, 4.0, 8.0, **TOL)
    ctx.check("finalize_logits", h, emb, 0.5, 8.0, **TOL)
    ctx.check("finalize_logits", h, emb, logit_scale=16.0, softcap=1.5, **TOL)


# Rejects scales_table_in_place.
@case("a tied head: the table that `embed` read is unchanged when the head uses it", public=True)
def tied_head(ctx):
    emb = table(ctx)
    tokens = ctx.randint(0, V, (2, 5))
    h = ctx.randn(2, 5, D)
    shared = ctx.fw.tensor(emb)  # one tensor, used for the lookup and as the head
    x = ctx.fw.numpy(ctx.solution.embed(ctx.fw.tensor(tokens), shared, 12.0))
    check.allclose(x, emb[tokens] * 12.0, label="embed", **TOL)
    check.allclose(
        ctx.fw.numpy(shared), emb, rtol=0.0, atol=0.0, label="the embedding table after `embed`"
    )
    logits = ctx.fw.numpy(ctx.solution.finalize_logits(ctx.fw.tensor(h), shared, 4.0))
    check.allclose(logits, ctx.expected("finalize_logits", h, emb, 4.0), label="logits", **TOL)


@case("token tensors of rank 0, 1, and 2, with repeated ids and the last id")
def token_shapes(ctx):
    emb = table(ctx)
    ctx.check("embed", np.array(V - 1), emb, 12.0, **TOL)
    ctx.check("embed", np.array([0, V - 1, 3, 3, 3]), emb, 2.5, **TOL)
    ctx.check("embed", np.array([[7, 7, 0, 1, 7], [V - 1, 0, 0, 2, 2]]), emb, 1.0, **TOL)
    ctx.check("embed", ctx.randint(0, V, (2, 5)), emb, -0.5, **TOL)


@case("hidden states of rank 1, 2, and 3")
def hidden_ranks(ctx):
    emb = table(ctx)
    for shape in ((D,), (5, D), (2, 5, D)):
        ctx.check("finalize_logits", ctx.randn(*shape), emb, 4.0, **TOL)
        ctx.check("finalize_logits", ctx.randn(*shape, scale=8.0), emb, 2.0, 5.0, **TOL)


@case("saturation: logits a hundred times the cap come out at plus or minus the cap")
def saturation(ctx):
    emb = table(ctx, scale=1.0)
    h = ctx.randn(5, D, scale=3000.0)
    got = ctx.check("finalize_logits", h, emb, 1.0, 30.0, **TOL)
    check.finite(got, label="finalize_logits")
    ctx.check("finalize_logits", h, emb, 100.0, 0.25, **TOL)


@case("small logits pass through the cap almost unchanged")
def small_logits(ctx):
    emb = table(ctx)
    h = ctx.randn(5, D, scale=0.05)
    got = ctx.check("finalize_logits", h, emb, 1.0, 30.0, rtol=1e-4, atol=1e-7)
    check.allclose(got, h @ emb.T, rtol=1e-3, atol=1e-6, label="capped logits far below the cap")


@case("random scales, divisors, and caps")
def random_sweep(ctx):
    for _ in range(6):
        emb = table(ctx, scale=float(10.0 ** ctx.rng.uniform(-1, 0.5)))
        ctx.check("embed", ctx.randint(0, V, (2, 5)), emb, float(ctx.rng.uniform(0.5, 50)), **TOL)
        h = ctx.randn(2, 5, D, scale=float(10.0 ** ctx.rng.uniform(-1, 1.5)))
        divisor = float(10.0 ** ctx.rng.uniform(-1, 1))
        cap = float(10.0 ** ctx.rng.uniform(-0.5, 1.5))
        ctx.check("finalize_logits", h, emb, divisor, **TOL)
        ctx.check("finalize_logits", h, emb, divisor, cap, **TOL)
