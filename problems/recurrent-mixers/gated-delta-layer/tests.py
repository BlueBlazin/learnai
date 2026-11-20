"""Tests for gated-delta-layer.

Weights come from the kit (`gdn_params`), expected values from three places: the
oracle reference, the kit's numpy layer (`gdn_layer_ref`), and, for the default
configuration, arrays computed by `Qwen3_5GatedDeltaNet` of `transformers`
(`fixtures/shipped.npz`, written by `fixtures/make.py`).

Every stateful case feeds the solution its own returned state. The chunkings include
chunks of one and two tokens, shorter than the K - 1 = 3 rows of convolution state.

JAX compiles once per input shape, so the sequence lengths are few: 12 tokens, cut
into pieces of 1, 2, 5, and 7.
"""

from dataclasses import replace

import numpy as np

from learnai.judge import case, check
from learnai.kit import GDNConfig, gdn_layer_ref, gdn_params

FN = "gdn_layer"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
CFG = GDNConfig()  # width 32, 2 key heads and 4 value heads of 8, kernel 4
STEPS = 12
TINY = GDNConfig(
    d_model=4, n_key_heads=1, n_value_heads=2, key_head_dim=2, value_head_dim=2, conv_kernel=2
)


def compare(got, want, what=""):
    """Compare (out, (conv_state, S)) with the expected triple, shapes first.

    `what` goes into the failure message: public cases pass one, hidden cases do not,
    because the message of a hidden case must not describe its input.
    """
    what = f"{what}: " if what else ""
    out, state = got
    check.that(
        isinstance(state, (tuple, list)) and len(state) == 2,
        f"{what}the second result is not a pair (conv_state, S)",
    )
    for value, expected, label in (
        (out, want[0], "out"),
        (state[0], want[1][0], "conv_state"),
        (state[1], want[1][1], "S"),
    ):
        check.shape(value, expected.shape, label=f"{what}{label}")
    check.allclose(out, want[0], **TOL, label=f"{what}out")
    check.allclose(state[0], want[1][0], **TOL, label=f"{what}conv_state")
    check.allclose(state[1], want[1][1], **TOL, label=f"{what}S")


def run_chunks(ctx, cfg, params, x, sizes, state=None):
    """Feed `x` in consecutive chunks, handing each call the state of the one before."""
    assert sum(sizes) == len(x)
    outs, start = [], 0
    for n in sizes:
        out, state = ctx.call(FN, params, cfg, x[start : start + n], state)
        check.shape(out, (n, cfg.d_model), label="out of one call")
        outs.append(out)
        start += n
    return np.concatenate(outs), state


@case("the worked example: width 4, kernel 2, three tokens, then the third one alone", public=True)
def worked_example(ctx):
    params = gdn_params(TINY, 3)
    x = np.array([[1, 0, -1, 2], [0.5, -0.5, 1, 0], [0, 2, 1, -1]], dtype=np.float32)
    out, state = ctx.call(FN, params, TINY, x, None)
    compare((out, state), ctx.expected(FN, params, TINY, x, None), "all three tokens")
    expected = [
        [-0.072025, 0.090774, 0.354554, 0.683884],
        [-0.109328, -0.223080, 0.181717, 0.175837],
        [-0.677092, -1.930008, 2.133805, 2.746412],
    ]
    check.allclose(out, np.array(expected, np.float32), **TOL, label="out")
    conv_state = [
        1.535249,
        -0.848628,
        -0.367922,
        0.590316,
        -0.777152,
        0.199583,
        -0.881968,
        1.245803,
    ]
    check.allclose(state[0], np.array([conv_state], np.float32), **TOL, label="conv_state")
    _, state = ctx.call(FN, params, TINY, x[:2], None)
    out, _ = ctx.call(FN, params, TINY, x[2:], state)
    check.allclose(out, np.array(expected[2:], np.float32), **TOL, label="third token, continued")


@case("12 tokens in one call, default configuration (2 key heads, 4 value heads)", public=True)
def one_pass(ctx):
    params, x = gdn_params(CFG, 1), ctx.randn(STEPS, CFG.d_model)
    got = ctx.call(FN, params, CFG, x, None)
    compare(got, ctx.expected(FN, params, CFG, x, None), "one call")
    compare(got, gdn_layer_ref(CFG, params, x), "one call, against the kit's layer")


@case("the same layer in Hugging Face Transformers: 12 tokens in one call", public=True)
def shipped_one_pass(ctx):
    f = ctx.fixture("shipped.npz")
    out, _state = ctx.call(FN, gdn_params(CFG, int(f["seed"])), CFG, f["x"], None)
    check.allclose(out, f["out_one_pass"], **TOL, label="out")


# Rejects conv_state_reset, memory_reset, conv_state_from_output, and (through the
# chunks of one and two tokens) conv_state_from_new_rows_only.
@case("any chunking of 12 tokens gives the outputs and the state of one call", public=True)
def chunking(ctx):
    params, x = gdn_params(CFG, 2), ctx.randn(STEPS, CFG.d_model)
    want = ctx.expected(FN, params, CFG, x, None)
    for sizes in ([5, 7], [1] * 12, [2, 5, 5], [1, 2, 2, 7]):
        compare(run_chunks(ctx, CFG, params, x, sizes), want, f"chunks {sizes}")


@case("the state after 1 and after 2 tokens of a new sequence", public=True)
def short_first_call(ctx):
    params, x = gdn_params(CFG, 3), ctx.randn(STEPS, CFG.d_model)
    for n in (1, 2):
        got = ctx.call(FN, params, CFG, x[:n], None)
        compare(got, ctx.expected(FN, params, CFG, x[:n], None), f"first {n} token(s)")
        projected = x[:n] @ params["in_proj_qkv.weight"].T
        rows = np.concatenate((np.zeros((3 - n, CFG.conv_dim), np.float32), projected))
        check.allclose(got[1][0], rows, **TOL, label="conv_state: zeros, then the projected inputs")


def silu(x):
    return x / (1.0 + np.exp(-x))


def layer_np(cfg, params, x, variant=None):
    """The layer on a new sequence in float64 numpy; `variant` names one wrong order."""
    p = {name: value.astype(np.float64) for name, value in params.items()}
    steps, r = len(x), cfg.n_value_heads // cfg.n_key_heads
    x = x.astype(np.float64)
    projected = x @ p["in_proj_qkv.weight"].T
    padded = np.concatenate((np.zeros((cfg.conv_kernel - 1, cfg.conv_dim)), projected))
    w = p["conv1d.weight"][:, 0]
    mixed = silu(sum(padded[j : j + steps] * w[:, j] for j in range(cfg.conv_kernel)))
    q, k, v = np.split(mixed, [cfg.key_dim, 2 * cfg.key_dim], axis=-1)

    def l2(u):
        u = u.reshape(steps, cfg.n_key_heads, cfg.key_head_dim)
        return np.repeat(u / np.sqrt((u * u).sum(-1, keepdims=True) + 1e-6), r, axis=1)

    q, k = l2(q), l2(k)
    if variant != "no_q_scale":
        q = q * cfg.key_head_dim**-0.5
    v = v.reshape(steps, cfg.n_value_heads, cfg.value_head_dim)
    g = np.exp(-np.exp(p["A_log"]) * np.logaddexp(x @ p["in_proj_a.weight"].T + p["dt_bias"], 0))
    beta = 1.0 / (1.0 + np.exp(-(x @ p["in_proj_b.weight"].T)))
    memory, ys = np.zeros((cfg.n_value_heads, cfg.value_head_dim, cfg.key_head_dim)), []
    for t in range(steps):
        memory = memory * g[t][:, None, None]
        delta = beta[t][:, None] * (v[t] - np.einsum("hvk,hk->hv", memory, k[t]))
        memory = memory + delta[:, :, None] * k[t][:, None, :]
        ys.append(np.einsum("hvk,hk->hv", memory, q[t]))
    y, gate = (
        np.stack(ys),
        silu((x @ p["in_proj_z.weight"].T).reshape(steps, cfg.n_value_heads, -1)),
    )
    if variant == "gate_before_norm":
        y, gate = y * gate, 1.0
    axes = (-2, -1) if variant == "norm_over_all_heads" else (-1,)
    y = p["norm.weight"] * y / np.sqrt((y * y).mean(axes, keepdims=True) + cfg.norm_eps) * gate
    return (y.reshape(steps, -1) @ p["out_proj.weight"].T).astype(np.float32)


# Written for gate_before_norm, no_q_scale, and norm_over_all_heads: each changes every
# output, so this case says which of them an output matches.
@case("norm per value head, then the gate; the query scaled by Dk^-0.5", public=True)
def norm_gate_scale(ctx):
    params, x = gdn_params(CFG, 9), ctx.randn(STEPS, CFG.d_model)
    out, _ = ctx.call(FN, params, CFG, x, None)
    check.shape(out, (STEPS, CFG.d_model), label="out")
    mistakes = {
        "gate_before_norm": "out is what a layer gives that applies the gate before the norm",
        "no_q_scale": "out is what a layer gives whose query is not multiplied by Dk^-0.5",
        "norm_over_all_heads": "out is what a layer gives whose RMS runs over all value heads",
    }
    for variant, message in mistakes.items():
        wrong = layer_np(CFG, params, x, variant)
        check.that(not np.allclose(out, wrong, **TOL), message)
    check.allclose(out, layer_np(CFG, params, x), **TOL, label="out")


# Rejects state_in_place: the caller keeps its state (to branch, to retry, to batch),
# so a call must not change the tensors it was given.
@case("the tensors passed in, the state included, are not modified", public=True)
def inputs_unchanged(ctx):
    params = gdn_params(CFG, 10)
    arrays = {
        "x": ctx.randn(5, CFG.d_model),
        "conv_state": ctx.randn(CFG.conv_kernel - 1, CFG.conv_dim),
        "S": ctx.randn(CFG.n_value_heads, CFG.value_head_dim, CFG.key_head_dim, scale=0.5),
    }
    tensors = {name: ctx.fw.tensor(value) for name, value in arrays.items()}
    weights = ctx.fw.to_framework(params)
    ctx.solution.gdn_layer(weights, CFG, tensors["x"], (tensors["conv_state"], tensors["S"]))
    for name, before in arrays.items():
        check.allclose(
            ctx.fw.numpy(tensors[name]), before, rtol=0, atol=0, label=f"{name} after the call"
        )
    for name, before in params.items():
        check.allclose(
            ctx.fw.numpy(weights[name]), before, rtol=0, atol=0, label=f"params[{name!r}]"
        )


# Rejects memory_transposed and both reset mutants without relying on the solution's
# own state: the state is drawn by the test.
@case("continuing from a state the test supplies")
def supplied_state(ctx):
    params = gdn_params(CFG, 4)
    state = (
        ctx.randn(CFG.conv_kernel - 1, CFG.conv_dim),
        ctx.randn(CFG.n_value_heads, CFG.value_head_dim, CFG.key_head_dim, scale=0.5),
    )
    for n in (1, 5):
        x = ctx.randn(n, CFG.d_model)
        got = ctx.call(FN, params, CFG, x, state)
        compare(got, ctx.expected(FN, params, CFG, x, state))


@case("the library with a cache: 5 tokens, then 7 calls of one token; outputs and final state")
def shipped_cached(ctx):
    f = ctx.fixture("shipped.npz")
    params = gdn_params(CFG, int(f["seed"]))
    got = run_chunks(ctx, CFG, params, f["x"], [5] + [1] * 7)
    compare(got, (f["out_cached"], (f["conv_state"], f["S"])))


# Rejects l2norm_eps_outside and no_q_scale decisively: with projections this small the
# squared norms of q and k are near 1e-6 and the outputs of the recurrence near 1e-4,
# so both epsilons and the Dk^-0.5 on the query decide the result.
@case("a quiet layer: in_proj_qkv 1000 times smaller")
def quiet_layer(ctx):
    params = dict(gdn_params(CFG, 5))
    params["in_proj_qkv.weight"] = params["in_proj_qkv.weight"] * np.float32(1e-3)
    x = ctx.randn(STEPS, CFG.d_model)
    want = ctx.expected(FN, params, CFG, x, None)
    compare(ctx.call(FN, params, CFG, x, None), want)
    compare(run_chunks(ctx, CFG, params, x, [5, 7]), want)


# One key head for four value heads, kernel 2 (one row of convolution state), and a
# norm_eps that is not the default.
@case("1 key head for 4 value heads, kernel 2, norm_eps = 1e-3")
def one_key_head(ctx):
    cfg = GDNConfig(
        d_model=16, n_key_heads=1, n_value_heads=4, key_head_dim=8, value_head_dim=6,
        conv_kernel=2, norm_eps=1e-3,
    )  # fmt: skip
    params, x = gdn_params(cfg, 7), ctx.randn(STEPS, 16)
    want = ctx.expected(FN, params, cfg, x, None)
    compare(ctx.call(FN, params, cfg, x, None), want)
    compare(run_chunks(ctx, cfg, params, x, [1] * 12), want)
    plain = replace(cfg, norm_eps=1e-6)
    check.that(
        not np.allclose(want[0], gdn_layer_ref(plain, params, x)[0], **TOL),
        "internal: norm_eps does not change this case",
    )


@case("a longer kernel: 6 taps, chunks shorter than its 5 rows of state")
def long_kernel(ctx):
    cfg = replace(CFG, conv_kernel=6)
    params, x = gdn_params(cfg, 8), ctx.randn(STEPS, cfg.d_model)
    want = ctx.expected(FN, params, cfg, x, None)
    compare(run_chunks(ctx, cfg, params, x, [2, 5, 5]), want)
    compare(run_chunks(ctx, cfg, params, x, [1, 2, 2, 7]), want)
