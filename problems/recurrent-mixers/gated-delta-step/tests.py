"""Tests for gated-delta-step.

Numeric cases compare with the oracle. Two cases compare with arrays computed by the
code `transformers` runs for one cached token of a Qwen3.5 delta layer
(`fixtures/shipped.npz`, written by `fixtures/make.py`). Property cases state what the
rule means without the reference: a write with full strength makes the memory return
the value under its key, and leaves every orthogonal key alone.

Shapes are shared across cases (JAX compiles once per shape): (H, Dv, Dk) is (3, 5, 4)
in most of them.
"""

import numpy as np

from learnai.judge import case, check

GATES, STEP = "gdn_gates", "gdn_step"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
H, DV, DK = 3, 5, 4


def f32(*values):
    return np.array(values, dtype=np.float32)


def unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def draw(ctx, h=H, dv=DV, dk=DK):
    """A random memory, query, key, value, and gates strictly inside (0, 1)."""
    return (
        ctx.randn(h, dv, dk),
        ctx.randn(h, dk, scale=0.5),
        ctx.randn(h, dk, scale=0.5),
        ctx.randn(h, dv),
        ctx.uniform(h, low=0.2, high=0.95),
        ctx.uniform(h, low=0.1, high=0.9),
    )


def step(ctx, *args):
    """Call the solution's step and check the shapes before anything reads the values."""
    y, memory = ctx.call(STEP, *args)
    check.shape(y, args[3].shape, label="y")
    check.shape(memory, args[0].shape, label="S")
    return y, memory


@case("the worked example: one head, a memory of 2 by 2, and two pairs of gates", public=True)
def worked_example(ctx):
    a, b = f32([0.0, 2.0]), f32([0.0, -1.0])  # one token, two heads
    g, beta = ctx.check(GATES, a, b, f32(0.0, -1.0), f32(0.0, 1.0), **TOL)
    check.allclose(g, f32([0.5, 0.325817]), atol=1e-5, label="g")
    check.allclose(beta, f32([0.5, 0.268941]), atol=1e-5, label="beta")

    memory = f32([[1.0, 0.0], [0.0, 2.0]])  # f32 adds the head axis: (1, 2, 2)
    q, k, v = f32([1.0, 0.0]), f32([0.6, 0.8]), f32([1.0, -1.0])
    y, new = ctx.check(STEP, memory, q, k, v, f32(0.5), f32(1.0), **TOL)
    check.allclose(y, f32([0.92, -1.08]), atol=1e-5, label="y with beta = 1")
    check.allclose(new, f32([[0.92, 0.56], [-1.08, -0.44]]), atol=1e-5, label="S with beta = 1")
    y, new = ctx.check(STEP, memory, q, k, v, f32(0.5), f32(0.5), **TOL)
    check.allclose(y, f32([0.71, -0.54]), atol=1e-5, label="y with beta = 0.5")
    check.allclose(new, f32([[0.71, 0.28], [-0.54, 0.28]]), atol=1e-5, label="S with beta = 0.5")


@case("one random step: 3 heads, values of width 5, keys of width 4", public=True)
def random_step(ctx):
    ctx.check(STEP, *draw(ctx), **TOL)


@case("gates for 6 tokens and 3 heads, raw inputs between -8 and 8", public=True)
def random_gates(ctx):
    a, b = ctx.uniform(6, H, low=-8.0, high=8.0), ctx.uniform(6, H, low=-8.0, high=8.0)
    a_log, dt_bias = np.log(f32(0.05, 0.7, 4.0)), f32(1.0, -0.5, 0.25)
    g, _beta = ctx.check(GATES, a, b, a_log, dt_bias, **TOL)
    check.that(
        bool((g >= 0).all() and (g <= 1).all()),
        "g: a decay outside [0, 1] came back",
        f"smallest {g.min()}, largest {g.max()}",
    )


# Rejects read_before_write, additive_write, decay_after_write, and
# correction_from_undecayed: each leaves S k different from v, or reads too early.
@case("g < 1, beta = 1, unit keys: the memory returns v under k, and y = v when q = k", public=True)
def full_write_lands(ctx):
    def prop(rng):
        memory, _q, k, v, g, _beta = draw(ctx)
        k = unit(k)
        y, new = step(ctx, memory, k, k, v, g, np.ones(H, np.float32))
        check.allclose(np.einsum("hvk,hk->hv", new, k), v, **TOL, label="S k after the step")
        check.allclose(y, v, **TOL, label="y for q = k")

    check.forall(prop, trials=5)


# Rejects additive_write on a nonzero memory and anything that touches more than the
# direction k.
@case("a write changes nothing the memory holds under a key orthogonal to k", public=True)
def orthogonal_keys_untouched(ctx):
    def prop(rng):
        memory, q, k, v, _g, beta = draw(ctx)
        other = ctx.randn(H, DK)
        other -= np.sum(other * k, -1, keepdims=True) / np.sum(k * k, -1, keepdims=True) * k
        _y, new = step(ctx, memory, q, k, v, np.ones(H, np.float32), beta)
        check.allclose(
            np.einsum("hvk,hk->hv", new, other),
            np.einsum("hvk,hk->hv", memory, other),
            **TOL,
            label="S k' for k' orthogonal to k, with g = 1",
        )

    check.forall(prop, trials=5)


# Rejects state_in_place: the caller keeps the memory it passed in.
@case("the tensors passed in, the memory included, are not modified", public=True)
def inputs_unchanged(ctx):
    arrays = draw(ctx)
    tensors = [ctx.fw.tensor(a) for a in arrays]
    ctx.solution.gdn_step(*tensors)
    for name, tensor, before in zip(
        ("S", "q", "k", "v", "g", "beta"), tensors, arrays, strict=True
    ):
        check.allclose(ctx.fw.numpy(tensor), before, rtol=0, atol=0, label=f"{name} after the call")


@case("an empty memory: the first token of a sequence")
def empty_memory(ctx):
    _memory, q, k, v, g, beta = draw(ctx)
    ctx.check(STEP, np.zeros((H, DV, DK), np.float32), q, k, v, g, beta, **TOL)


# Square everything: a gate broadcast along the wrong axis still has a legal shape here.
@case("heads, value width, and key width all 4")
def square_shapes(ctx):
    for _ in range(3):
        ctx.check(STEP, *draw(ctx, 4, 4, 4), **TOL)


@case("one head")
def one_head(ctx):
    ctx.check(STEP, *draw(ctx, 1, DV, DK), **TOL)


# Rejects normalizes_key: the layer normalizes keys before the step, the step does not.
@case("long keys and queries are used as given")
def keys_as_given(ctx):
    memory, q, k, v, g, beta = draw(ctx)
    ctx.check(STEP, memory, 3.0 * q, 2.0 * k, v, g, beta, **TOL)


# Rejects beta_scales_value_only (S + (beta v - S k) k^T) and the decay-order mutants
# at the two ends of each gate.
@case("gates at their extremes: g = 0 or 1, beta = 0 or 1")
def extreme_gates(ctx):
    memory, q, k, v, _g, _beta = draw(ctx)
    for g, beta in ((1.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), (1e-3, 1.0), (1.0, 0.3)):
        gates = np.full(H, g, np.float32), np.full(H, beta, np.float32)
        ctx.check(STEP, memory, q, k, v, *gates, **TOL)


# The library's own numbers: gates as Qwen3_5GatedDeltaNet computes them, then six
# tokens of torch_recurrent_gated_delta_rule from a nonzero memory.
@case("six tokens in a row agree with the delta rule of Hugging Face Transformers")
def shipped_sequence(ctx):
    f = ctx.fixture("shipped.npz")
    g, beta = ctx.call(GATES, f["a"], f["b"], f["A_log"], f["dt_bias"])
    check.allclose(g, f["g"], **TOL, label="g")
    check.allclose(beta, f["beta"], **TOL, label="beta")
    memory = f["S0"]
    for t in range(len(f["q"])):
        y, memory = step(ctx, memory, f["q"][t], f["k"][t], f["v"][t], f["g"][t], f["beta"][t])
        check.allclose(y, f["y"][t], **TOL, label="y")
    check.allclose(memory, f["S"], **TOL, label="S")


# Rejects no_softplus (g > 1 for negative a) and bias_outside_softplus.
@case("gates where a + dt_bias is very negative or very positive")
def gate_tails(ctx):
    a = f32([-20.0, -12.0, -3.0], [0.0, 3.0, 20.0], [-1.0, 1.0, 9.0], [5.0, -5.0, 0.5])
    b = f32([-20.0, 0.0, 20.0], [1.0, -1.0, 3.0], [-6.0, 6.0, 0.0], [0.5, 2.0, -2.0])
    for dt_bias in (f32(0.0, 0.0, 0.0), f32(-1.0, 0.5, 1.0)):
        g, _beta = ctx.check(GATES, a, b, np.log(f32(0.01, 1.0, 16.0)), dt_bias, **TOL)
        check.that(
            bool((g >= 0).all() and (g <= 1).all()),
            "g: a decay outside [0, 1] came back",
            f"smallest {g.min()}, largest {g.max()}",
        )


@case("gates for a single token")
def one_token_gates(ctx):
    a, b = ctx.uniform(1, H, low=-4.0, high=4.0), ctx.uniform(1, H, low=-4.0, high=4.0)
    ctx.check(GATES, a, b, np.log(f32(0.05, 0.7, 4.0)), f32(1.0, -0.5, 0.25), **TOL)
