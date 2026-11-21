"""Tests for per-layer-embeddings.

Numeric cases compare with the oracle. Three cases compare with arrays computed by
`Gemma4TextModel` and `Gemma4TextDecoderLayer` of `transformers` on a tiny configuration
(`fixtures/shipped.npz`, written by `fixtures/make.py`), so a formula the reference got
wrong would fail there too.

Shapes are few (JAX compiles once per shape): most cases use 5 tokens, width 16,
3 layers, per-layer width 4, a vocabulary of 11, and an MLP width of 24.
"""

import numpy as np

from learnai.judge import case, check

INPUTS, INJECT, LAYER = "ple_inputs", "ple_inject", "ple_layer"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
T, D, L, P, V, M = 5, 16, 3, 4, 11, 24
NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "post_per_layer_input_norm.weight",
)


def draw_model(ctx, steps=T, dim=D, layers=L, d_ple=P, vocab=V, h0_scale=1.0):
    """(tokens, h0, table, w_model, norm_gain): the arguments of ple_inputs but n_layers."""
    return (
        ctx.randint(0, vocab, (steps,)),
        ctx.randn(steps, dim, scale=h0_scale),
        ctx.randn(vocab, layers * d_ple),
        ctx.randn(layers * d_ple, dim, scale=dim**-0.5),
        ctx.uniform(d_ple, low=0.5, high=1.5),
    )


def draw_layer(ctx, dim=D, d_ple=P, inter=M, gate_scale=1.0):
    """The weights of one layer, by name. Norm gains in [0.5, 1.5], layer_scalar not 1."""
    params = {name: ctx.uniform(dim, low=0.5, high=1.5) for name in NORMS}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        params[f"self_attn.{name}.weight"] = ctx.randn(dim, dim, scale=dim**-0.5)
    params["mlp.gate_proj.weight"] = ctx.randn(inter, dim, scale=dim**-0.5)
    params["mlp.up_proj.weight"] = ctx.randn(inter, dim, scale=dim**-0.5)
    params["mlp.down_proj.weight"] = ctx.randn(dim, inter, scale=inter**-0.5)
    params["per_layer_input_gate.weight"] = ctx.randn(d_ple, dim, scale=gate_scale * dim**-0.5)
    params["per_layer_projection.weight"] = ctx.randn(dim, d_ple, scale=d_ple**-0.5)
    params["layer_scalar"] = ctx.uniform(1, low=0.6, high=0.9)
    return params


def inject_args(params):
    return (
        params["per_layer_input_gate.weight"],
        params["per_layer_projection.weight"],
        params["post_per_layer_input_norm.weight"],
    )


def causal(steps):
    return np.tril(np.ones((steps, steps), dtype=bool))


def shipped_layer(f, i):
    prefix = f"layer{i}."
    return {k[len(prefix) :]: f[k] for k in f if k.startswith(prefix)}


@case("the worked example: one token, two layers of width 2", public=True)
def worked_example(ctx):
    tokens = np.array([1])
    h0 = np.array([[2.0, 0.0]], dtype=np.float32)
    table = np.array([[9, 9, 9, 9], [1, 0, 0, 2], [9, 9, 9, 9]], dtype=np.float32)
    w_model = np.array([[1, 0], [1, 0], [0, 1], [3, 0]], dtype=np.float32)
    ones = np.ones(2, dtype=np.float32)
    ple = ctx.check(INPUTS, tokens, h0, table, w_model, ones, 2, **TOL)
    want = np.array([[[1.707106, 0.707106], [0.0, 3.0]]], dtype=np.float32)
    check.allclose(ple, want, atol=1e-5, label=INPUTS)

    h = np.array([[1.0, -1.0]], dtype=np.float32)
    eye = np.eye(2, dtype=np.float32)
    got = ctx.check(INJECT, h, want[:, 0], eye, eye, ones, **TOL)
    check.allclose(got, np.array([[2.409909, -1.110254]], np.float32), atol=1e-5, label=INJECT)


@case("ple_inputs against Gemma4TextModel, for h0 of order 1 and of order 1e-3", public=True)
def shipped_inputs(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.call(INPUTS, f["tokens"], f["h0"], f["table"], f["w_model"], f["norm_gain"], L)
    check.allclose(got, f["ple"], **TOL, label=INPUTS, dtype=True)
    # The same call with h0 / 1000. The projected part's mean square is then near the
    # epsilon of its norm, which is where the factor D ** -0.5 in front of that norm
    # (and the value of the epsilon) shows.
    got = ctx.call(INPUTS, f["tokens"], f["h0_small"], f["table"], f["w_model"], f["norm_gain"], L)
    check.allclose(got, f["ple_small"], **TOL, label=f"{INPUTS} with h0 / 1000")


@case("ple_inject and ple_layer against Gemma4TextDecoderLayer, layers 0 to 2", public=True)
def shipped_layers(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.call(INJECT, f["inject_h"], f["inject_p"], *inject_args(shipped_layer(f, 0)))
    check.allclose(got, f["inject_y"], **TOL, label=INJECT, dtype=True)
    for i in range(L):
        got = ctx.call(LAYER, shipped_layer(f, i), f["layer_x"], f["ple"], i, causal(T))
        check.allclose(got, f[f"layer{i}_y"], **TOL, label=f"{LAYER}, layer {i}", dtype=True)


@case("layer i reads only columns i * d_ple to (i + 1) * d_ple of the packed row", public=True)
def layer_reads_own_columns(ctx):
    def prop(rng):
        tokens, h0, table, w_model, gain = draw_model(ctx)
        i = int(rng.integers(0, L))
        own = np.zeros(L * P, dtype=bool)
        own[i * P : (i + 1) * P] = True
        base = ctx.call(INPUTS, tokens, h0, table, w_model, gain, L)
        check.shape(base, (T, L, P), label=INPUTS)

        table2, w2 = table.copy(), w_model.copy()
        table2[:, ~own] = ctx.randn(V, L * P - P)
        w2[~own] = ctx.randn(L * P - P, D, scale=D**-0.5)
        other = ctx.call(INPUTS, tokens, h0, table2, w2, gain, L)
        check.allclose(
            other[:, i],
            base[:, i],
            rtol=1e-6,
            atol=1e-6,
            label=f"layer {i} of the result after changing only the other layers' columns",
        )
        table3 = table.copy()
        table3[:, own] += 1.0
        moved = ctx.call(INPUTS, tokens, h0, table3, w_model, gain, L)
        check.that(
            bool((np.abs(moved[:, i] - base[:, i]) > 1e-3).all()),
            f"{INPUTS}: layer {i} of the result ignores some of its own table columns",
            f"columns {i * P} to {(i + 1) * P - 1} of every table row were raised by 1",
        )

    check.forall(prop, trials=4)


@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    model = dict(
        zip(("tokens", "h0", "table", "w_model", "norm_gain"), draw_model(ctx), strict=True)
    )
    params = draw_layer(ctx)
    ple = ctx.expected(INPUTS, *model.values(), L)
    gate, proj, gain = inject_args(params)
    inject = {
        "h": ctx.randn(T, D),
        "ple_i": ple[:, 1],
        "w_gate": gate,
        "w_proj": proj,
        "post_gain": gain,
    }
    layer = {"x": ctx.randn(T, D), "ple": ple, "mask": causal(T), **params}

    def watch(fn, arrays, call):
        tensors = {name: ctx.fw.tensor(a) for name, a in arrays.items()}
        call(tensors)
        for name, before in arrays.items():
            after = ctx.fw.numpy(tensors[name])
            if before.dtype == bool:
                check.equal(after, before, label=f"{name} after the call to {fn}")
            else:
                check.allclose(
                    after, before, rtol=0, atol=0, label=f"{name} after the call to {fn}"
                )

    watch(INPUTS, model, lambda t: ctx.solution.ple_inputs(*t.values(), L))
    watch(INJECT, inject, lambda t: ctx.solution.ple_inject(*t.values()))
    watch(
        LAYER,
        layer,
        lambda t: ctx.solution.ple_layer({k: t[k] for k in params}, t["x"], t["ple"], 1, t["mask"]),
    )


@case("random inputs: all three functions")
def random_inputs(ctx):
    for _ in range(2):
        model = draw_model(ctx)
        ple = ctx.check(INPUTS, *model, L, **TOL)
        params = draw_layer(ctx)
        ctx.check(INJECT, ctx.randn(T, D), ple[:, 2], *inject_args(params), **TOL)
        ctx.check(LAYER, params, ctx.randn(T, D), ple, 0, causal(T), **TOL)


# Rejects layer_zero_slice. Layer 0 is covered above; here the slice must be i's.
@case("the layers after the first")
def later_layers(ctx):
    model = draw_model(ctx)
    ple = ctx.expected(INPUTS, *model, L)
    x = ctx.randn(T, D)
    for i in (1, 2):
        ctx.check(LAYER, draw_layer(ctx), x, ple, i, causal(T), **TOL)


# Rejects context_scale_omitted and default_eps: with a projection whose mean square is
# near 1e-6, the epsilon inside the root is no longer negligible, so both the factor
# D ** -0.5 in front of the norm and the value of the epsilon show in the result.
@case("a small layer-0 input")
def small_input(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.call(INPUTS, f["tokens"], f["h0_small"], f["table"], f["w_model"], f["norm_gain"], L)
    check.allclose(got, f["ple_small"], **TOL, label=INPUTS)
    for scale in (3e-3, 1e-3):
        ctx.check(INPUTS, *draw_model(ctx, h0_scale=scale), L, **TOL)
    # The norms of the layer have the same epsilon: a stream and branches of order 1e-3.
    params = draw_layer(ctx)
    model = draw_model(ctx)
    ple = ctx.expected(INPUTS, *model, L)
    ctx.check(INJECT, ctx.randn(T, D, scale=2e-3), ple[:, 0] * 0.5, *inject_args(params), **TOL)
    for name in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        params[name] = params[name] * 2e-3
    ctx.check(LAYER, params, ctx.randn(T, D, scale=2e-3), ple, 1, causal(T), **TOL)


# As many layers as per-layer channels: a row cut as (d_ple, L) has a legal shape here.
@case("square: 4 layers of per-layer width 4")
def square_layers(ctx):
    model = draw_model(ctx, layers=4)
    ple = ctx.check(INPUTS, *model, 4, **TOL)
    ctx.check(LAYER, draw_layer(ctx), ctx.randn(T, D), ple, 3, causal(T), **TOL)


# Rejects no_gate_activation and exact_gelu: gate pre-activations of order 3, where the
# activation is far from linear and the two GELU formulas are furthest apart.
@case("large gate pre-activations")
def large_gates(ctx):
    model = draw_model(ctx)
    ple = ctx.expected(INPUTS, *model, L)
    for _ in range(2):
        params = draw_layer(ctx, gate_scale=3.0)
        ctx.check(INJECT, ctx.randn(T, D), ple[:, 1], *inject_args(params), **TOL)
        ctx.check(LAYER, params, ctx.randn(T, D), ple, 2, causal(T), **TOL)


@case("one token")
def one_token(ctx):
    model = draw_model(ctx, steps=1)
    ple = ctx.check(INPUTS, *model, L, **TOL)
    params = draw_layer(ctx)
    ctx.check(INJECT, ctx.randn(1, D), ple[:, 0], *inject_args(params), **TOL)
    ctx.check(LAYER, params, ctx.randn(1, D), ple, 2, causal(1), **TOL)


@case("a mask that is not causal: each token sees itself and one token back")
def window_mask(ctx):
    model = draw_model(ctx)
    ple = ctx.expected(INPUTS, *model, L)
    mask = causal(T) & ~np.tril(np.ones((T, T), dtype=bool), -2)
    ctx.check(LAYER, draw_layer(ctx), ctx.randn(T, D), ple, 1, mask, **TOL)


@case("35 layers of per-layer width 8, a repeated token")
def many_layers(ctx):
    tokens, h0, table, w_model, gain = draw_model(ctx, dim=48, layers=35, d_ple=8)
    tokens[3] = tokens[0]
    ple = ctx.check(INPUTS, tokens, h0, table, w_model, gain, 35, **TOL)
    params = draw_layer(ctx, dim=48, d_ple=8, inter=64)
    ctx.check(LAYER, params, ctx.randn(T, 48), ple, 34, causal(T), **TOL)
