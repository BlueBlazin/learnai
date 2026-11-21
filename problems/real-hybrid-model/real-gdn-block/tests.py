"""Tests for real-gdn-block.

Two sources of expected values, neither of them the reference solution:

- The cases marked `assets=[]` run a four-layer model with random weights in the
  checkpoint's layout (`checkpoint_layout(hybrid_params(...))`: the real names, block
  norms stored as gain - 1). It is 32 wide with 4 key heads of 4 and 4 value heads of
  8, so the key and value widths differ. Expected outputs and states come from the
  kit's numpy block, `hybrid_block_ref`, on the same weights in the kit's layout. The
  kit's unit tests hold that block against `transformers` on such models.
- The other cases run layers 1 and 13 of the real Qwen3.5 0.8B on the hidden states of
  a 12-token prompt. Inputs and expected outputs are hidden states computed by
  `transformers` in one call (`fixtures/make.py`), compared at 64 fixed columns.

JAX compiles once per input shape, so the shapes are few: the small model sees 6 rows,
cut 4 + 2, 2 + 2 + 2, or 4 + 1 + 1 (a prefill and two decode steps); the real one 12
rows, cut 5 + 7.

Nothing on disk holds the answer of a synthetic case: inputs are drawn here and the
expected values are computed by the kit while the case runs. (The real-weight fixture
holds 64 of 1024 output columns, so a solution that replayed it would still fail every
synthetic case.)
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import HybridConfig, RealHybrid, checkpoint_layout, hybrid_block_ref, hybrid_params

FN = "qwen_gdn_block"
ASSET = "qwen3.5-0.8b"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TOL = {"rtol": 1e-3, "atol": 1e-5}  # hidden states; stated in Constraints
STATE_TOL = {"rtol": 1e-3, "atol": 1e-4}  # conv_state and S

# The small model: layers 0, 1, 2 are linear attention, layer 3 is attention (unused).
CFG = HybridConfig(n_key_heads=4, n_value_heads=4, key_head_dim=4, value_head_dim=8)
SEED = 250
STEPS = 6
# y[5, :4] of the worked example, as printed in the statement.
WORKED = [0.775550, -0.210093, -0.297908, 0.922919]
_cache: dict[str, object] = {}


def small():
    """(kit-layout params, checkpoint-layout weights) of the small model."""
    if "small" not in _cache:
        params = hybrid_params(CFG, SEED)
        _cache["small"] = (params, checkpoint_layout(params))
    return _cache["small"]


def rows(seed: int, steps: int = STEPS) -> np.ndarray:
    """Seeded input rows for the small model, (steps, 32)."""
    return np.random.default_rng(seed).standard_normal((steps, CFG.d_model)).astype(np.float32)


def unpack(result, label):
    """(x, conv_state, S) of a result, after checking that it is (x, (conv_state, S))."""
    ok = isinstance(result, (tuple, list)) and len(result) == 2
    ok = ok and isinstance(result[1], (tuple, list)) and len(result[1]) == 2
    check.that(ok, f"{label}: must return (x, (conv_state, S))")
    return result[0], result[1][0], result[1][1]


def compare(got, want, label, *, state=True):
    x, conv_state, memory = unpack(got, label)
    check.shape(x, want[0].shape, label=f"{label}: x")
    check.allclose(x, want[0], **TOL, label=f"{label}: x", dtype=True)
    if state:
        for value, expected, name in ((conv_state, want[1][0], "conv_state"),
                                      (memory, want[1][1], "S")):  # fmt: skip
            check.shape(value, expected.shape, label=f"{label}: {name}")
            check.allclose(value, expected, **STATE_TOL, label=f"{label}: {name}", dtype=True)


def in_pieces(ctx, x, weights, layer, pieces, label, state=None):
    """Feed `x` in consecutive calls, each with the state the previous one returned."""
    outs, start = [], 0
    for size in pieces:
        result = ctx.call(FN, x[start : start + size], weights, layer, state)
        out, conv_state, memory = unpack(result, label)
        check.shape(out, (size, x.shape[1]), label=f"{label}: x of a {size}-row call")
        outs.append(out)
        state = (conv_state, memory)
        start += size
    return np.concatenate(outs), state


# ---------------------------------------------------------------------------------
# The small random model: no download needed.
# ---------------------------------------------------------------------------------


# Rejects no_one_plus, one_plus_on_gated_norm, conv_kernel_flipped, dt_bias_omitted,
# computes_in_bfloat16, mlp_reads_block_input, gate_before_norm, and swiglu_swapped: a
# wrong formula moves every row (dt_bias and A_log: every row but the first).
@case("the worked example: layer 1 of the small model, 6 rows at once, then 4 + 2",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    params, weights = small()
    x = rows(1)
    want = hybrid_block_ref(CFG, params, 1, x)
    label = "qwen_gdn_block(x, weights, 1, None)"
    got = ctx.call(FN, x, weights, 1, None)
    compare(got, want, label)
    check.allclose(got[0][5, :4], WORKED, **TOL, label=f"{label}: x[5, :4]")
    # The same rows in two calls: the state of the first is the fourth argument of the second.
    out, state = in_pieces(ctx, x, weights, 1, (4, 2), "4 rows, then 2 with the returned state")
    compare((out, state), want, "4 rows, then 2 with the returned state")


# Other layers and inputs; also the first comparison of the returned state. With state
# None the first row has no memory to decay, so the decay constants do not touch it.
@case("layers 0 and 2 of the small model, with the returned conv_state and S",
      public=True, assets=[])  # fmt: skip
def other_layers(ctx):
    params, weights = small()
    for layer, seed in ((0, 2), (2, 3)):
        x = rows(seed)
        want = hybrid_block_ref(CFG, params, layer, x)
        compare(ctx.call(FN, x, weights, layer, None), want,
                f"qwen_gdn_block(x, weights, {layer}, None)")  # fmt: skip


# Rejects ignores_state and conv_state_restarts: the state handed in is the kit's, so a
# solution with a private state layout that only agrees with itself fails here too.
@case("continuing from a given state: 2 more rows after 4, and 3 calls of 2 rows",
      public=True, assets=[])  # fmt: skip
def continues_from_state(ctx):
    params, weights = small()
    x = rows(4)
    want = hybrid_block_ref(CFG, params, 0, x)
    _, before = hybrid_block_ref(CFG, params, 0, x[:4])
    got = ctx.call(FN, x[4:], weights, 0, before)
    compare(got, (want[0][4:], want[1]), "qwen_gdn_block(x[4:], weights, 0, state after x[:4])")
    # Calls shorter than the 3 rows of convolution state.
    out, state = in_pieces(ctx, x, weights, 0, (2, 2, 2), "6 rows in calls of 2")
    compare((out, state), want, "6 rows in calls of 2")


# Rejects norm_eps_default without the download: at this scale mean(x^2) is near 1e-3,
# as in the real model, and 1e-5 in place of 1e-6 moves the norm by half a percent.
@case("inputs as small as the real model's hidden states (about 0.03): the norm epsilon counts",
      public=True, assets=[])  # fmt: skip
def small_inputs(ctx):
    params, weights = small()
    x = rows(5) * np.float32(0.03)
    want = hybrid_block_ref(CFG, params, 2, x)
    compare(ctx.call(FN, x, weights, 2, None), want, "qwen_gdn_block(0.03 * x, weights, 2, None)")


def check_untouched(ctx, layer, x, label):
    """Continue from a state after x[:4] with x[4:], calling the solution directly on
    tensors this test keeps, and check that x, conv_state, and S still hold their values."""
    params, weights = small()
    key = f"small_{ctx.framework}"
    if key not in _cache:
        _cache[key] = {name: ctx.fw.tensor(value) for name, value in weights.items()}
    _, before = hybrid_block_ref(CFG, params, layer, x[:4])
    originals = {"x": x[4:], "conv_state": before[0], "S": before[1]}
    # Copies, so that a tensor sharing memory with a numpy array cannot change the original.
    held = {name: ctx.fw.tensor(np.array(value)) for name, value in originals.items()}
    result = ctx.solution.qwen_gdn_block(
        held["x"], _cache[key], layer, (held["conv_state"], held["S"])
    )
    for name, original in originals.items():
        now = np.asarray(ctx.fw.to_numpy(held[name]))
        check.that(
            now.shape == original.shape and np.array_equal(now, original),
            f"{label}: the argument `{name}` was modified in place",
            "a caller may continue twice from one state, so the tensors passed in must keep "
            "their values; build new tensors instead of updating these",
        )
    out, conv_state, memory = unpack(result, label)
    got = tuple(np.asarray(ctx.fw.to_numpy(t)) for t in (out, conv_state, memory))
    want = hybrid_block_ref(CFG, params, layer, x[4:], before)
    compare((got[0], (got[1], got[2])), want, label)


# Rejects state_updated_in_place: `ctx.call` hands the solution fresh tensors, so only
# a direct call on tensors the test keeps can see an in-place update.
@case("the tensors passed in keep their values: x, conv_state, and S after a continued call",
      public=True, assets=[])  # fmt: skip
def arguments_untouched(ctx):
    check_untouched(ctx, 1, rows(6), "qwen_gdn_block(x[4:], weights, 1, (conv_state, S))")


# The decode step of a server: one row per call, after a prefill.
@case("a prefill of 4 rows, then two decode steps of one row each",
      public=True, assets=[])  # fmt: skip
def decode_steps(ctx):
    params, weights = small()
    x = rows(7)
    want = hybrid_block_ref(CFG, params, 1, x)
    out, state = in_pieces(ctx, x, weights, 1, (4, 1, 1), "4 rows, then 1, then 1")
    compare((out, state), want, "4 rows, then 1, then 1")


@case("the small model: other inputs, larger and smaller", assets=[])
def other_inputs(ctx):
    params, weights = small()
    for layer, scale in ((2, 4.0), (1, 0.25)):
        x = ctx.randn(STEPS, CFG.d_model, scale=scale)
        want = hybrid_block_ref(CFG, params, layer, x)
        compare(ctx.call(FN, x, weights, layer, None), want, FN)


@case("the small model: a state carried over three calls on another layer", assets=[])
def carried_state(ctx):
    params, weights = small()
    x = ctx.randn(STEPS, CFG.d_model)
    want = hybrid_block_ref(CFG, params, 2, x)
    out, state = in_pieces(ctx, x, weights, 2, (2, 2, 2), FN)
    compare((out, state), want, FN)
    _, before = hybrid_block_ref(CFG, params, 1, x[:2])
    want = hybrid_block_ref(CFG, params, 1, x[2:], before)
    compare(ctx.call(FN, x[2:], weights, 1, before), want, FN)


@case("the small model: one-row calls on another layer, and arguments left as they were",
      assets=[])  # fmt: skip
def decode_and_untouched(ctx):
    params, weights = small()
    x = ctx.randn(STEPS, CFG.d_model)
    want = hybrid_block_ref(CFG, params, 0, x)
    out, state = in_pieces(ctx, x, weights, 0, (4, 1, 1), FN)
    compare((out, state), want, FN)
    check_untouched(ctx, 2, ctx.randn(STEPS, CFG.d_model, scale=2.0), FN)


# ---------------------------------------------------------------------------------
# Qwen3.5 0.8B: width 1024, 16 key and 16 value heads of 128, kernel 4, SwiGLU 3584.
# ---------------------------------------------------------------------------------


def real(ctx):
    """(framework weights, fixture). A tensor is decoded the first time it is read."""
    fixture = ctx.fixture("qwen3.5-0.8b.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    key = f"weights_{ctx.framework}"
    if key not in _cache:
        # One Weights per framework for the run: a layer is decoded from bfloat16 once.
        _cache[key] = RealHybrid(ctx.asset(ASSET)).weights(ctx.fw)
    return _cache[key], fixture


def run_real(ctx, layer, pieces, what):
    """Run the block on the fixture's input in consecutive calls; compare 64 columns."""
    weights, fixture = real(ctx)
    x = fixture[f"x_{layer}"]
    outs, state, start = [], None, 0
    for size in pieces:
        result = ctx.solution.qwen_gdn_block(
            ctx.fw.tensor(x[start : start + size]), weights, layer, state
        )
        out, conv_state, memory = unpack(result, what)
        check.that(
            ctx.fw.is_tensor(out),
            f"{what}: x is a {type(out).__name__}, not a tensor of the framework",
        )
        out = np.asarray(ctx.fw.to_numpy(out))
        check.shape(out, (size, 1024), label=f"{what}: x")
        check.that(out.dtype == np.float32, f"{what}: x has dtype {out.dtype}, not float32")
        outs.append(out)
        state = (conv_state, memory)
        start += size
    got = np.concatenate(outs)
    check.allclose(got[:, fixture["columns"]], fixture[f"y_{layer}"], **TOL,
                   label=f"{what}: x at 64 columns")  # fmt: skip
    conv_state, memory = (np.asarray(ctx.fw.to_numpy(part)) for part in state)
    check.shape(conv_state, (3, 6144), label=f"{what}: conv_state")
    check.shape(memory, (16, 128, 128), label=f"{what}: S")


# Rejects layer_prefix_without_dot (only a model with more than ten layers has names
# that start with another layer's), the one mutant no small model catches. Also rejects
# norm_eps_default: the real hidden states are small enough for the epsilon to count.
@case("Qwen3.5 0.8B, layer 1: the 12 hidden states of a prompt, in one call", public=True)
def real_layer_1(ctx):
    run_real(ctx, 1, (12,), "qwen_gdn_block(x, weights, 1, None)")


@case("Qwen3.5 0.8B, layer 13: 5 rows, then 7 with the returned state", public=True)
def real_layer_13_two_calls(ctx):
    run_real(ctx, 13, (5, 7), "layer 13 in two calls")


@case("Qwen3.5 0.8B, layer 13 in one call")
def real_layer_13(ctx):
    run_real(ctx, 13, (12,), FN)


@case("Qwen3.5 0.8B, layer 1 in two calls")
def real_layer_1_two_calls(ctx):
    run_real(ctx, 1, (5, 7), FN)
