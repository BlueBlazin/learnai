"""Tests for mamba2-ssd-step.

Numeric cases compare with the oracle. Two cases compare with arrays computed by the
Mamba-2 functions of `transformers` for Nemotron (`fixtures/shipped.npz`, written by
`fixtures/make.py`): its one-token update, and its multi-token scan with the clamp of
the shipped configuration. The property cases tie the two entry points together: the
matrix applied to a whole sequence equals the step loop from a zero state.

Shapes are shared (JAX compiles once per shape): 4 heads of width 3 in 2 groups with
state size 5, and sequences of 9 tokens for the matrix.
"""

import numpy as np

from learnai.judge import case, check

STEP, MATRIX = "ssd_step", "ssd_matrix"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
H, P, N, G, T = 4, 3, 5, 2, 9
INF = float("inf")
NEMOTRON = (0.001, INF)  # (time_step_min, inf): what transformers passes for Nemotron


def f32(*values):
    return np.array(values, dtype=np.float32)


def softplus(x):
    x = np.asarray(x, np.float64)
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def params(ctx, heads=H):
    """A_log, D, dt_bias: rates between 0.3 and 4, skips and biases of either sign."""
    return (
        np.log(ctx.uniform(heads, low=0.3, high=4.0)),
        ctx.uniform(heads, low=-1.5, high=1.5),
        ctx.uniform(heads, low=-1.0, high=1.0),
    )


def step(ctx, h, x, dt_raw, b, c, a_log, d, dt_bias, limit, **tol):
    return ctx.check(STEP, h, x, dt_raw, b, c, a_log, d, dt_bias, limit, **(tol or TOL))


def loop(ctx, x, dt_raw, b, c, a_log, d, dt_bias, limit, h=None):
    """Run the solution's step over a sequence; returns (T, H, P) outputs and the state."""
    h = np.zeros((x.shape[1], x.shape[2], b.shape[2]), np.float32) if h is None else h
    ys = []
    for t in range(len(x)):
        y, h = ctx.call(STEP, h, x[t], dt_raw[t], b[t], c[t], a_log, d, dt_bias, limit)
        check.shape(y, x[t].shape, label="y")
        ys.append(y)
    return np.stack(ys), h


@case("the worked example: one head, a state of size 2, and a 3-token matrix", public=True)
def worked_example(ctx):
    h, x = f32([[1.0, 2.0]]), f32([2.0])
    b, c = f32([1.0, -1.0]), f32([0.5, 0.25])
    args = (b, c, f32(0.0), f32(3.0), f32(0.0), NEMOTRON)
    y, new = step(ctx, h, x, f32(0.0), *args)
    check.allclose(y, f32([6.846574]), **TOL, label="y")
    check.allclose(new, f32([[1.886294, -0.386294]]), **TOL, label="h")
    y, new = step(ctx, h, x, f32(-10.0), *args)
    check.allclose(y, f32([6.999500]), **TOL, label="y with the clamp")
    check.allclose(new, f32([[1.001000, 1.996002]]), **TOL, label="h with the clamp")

    dt = f32([0.5], [1.0], [2.0])
    b = f32([[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]])
    c = f32([[1.0, 1.0]], [[2.0, 1.0]], [[1.0, -0.5]])
    m = ctx.check(MATRIX, dt, b, c, np.log(np.log(f32(2.0))), **TOL)
    want = f32([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0625, -0.125, 1.0]])
    check.allclose(m, want, **TOL, label="M")


@case("one random step: 4 heads of width 3 in 2 groups, state size 5", public=True)
def random_step(ctx):
    a_log, d, dt_bias = params(ctx)
    for limit in (NEMOTRON, (0.0, INF)):
        step(ctx, ctx.randn(H, P, N), ctx.randn(H, P), ctx.randn(H, scale=2.0),
             ctx.randn(G, N), ctx.randn(G, N), a_log, d, dt_bias, limit)  # fmt: skip


# Rejects no_clamp and softplus_missing: softplus gives 4.5e-5 to 6e-9 here, far below
# the limit, so the step size is the limit itself.
@case("raw step sizes of -10 to -19: softplus falls below dt_limit[0] = 0.001", public=True)
def lower_clamp(ctx):
    a_log, d, dt_bias = params(ctx)
    dt_raw = f32(-10.0, -13.0, -16.0, -19.0) - dt_bias
    step(ctx, ctx.randn(H, P, N), ctx.randn(H, P), dt_raw,
         ctx.randn(G, N), ctx.randn(G, N), a_log, d, dt_bias, NEMOTRON)  # fmt: skip


# The library's one-token function, six tokens in a row from a nonzero state.
@case("six cached tokens agree with the one-token update of Hugging Face Transformers", public=True)
def shipped_steps(ctx):
    f = ctx.fixture("shipped.npz")
    args = (f["A_log"], f["D"], f["dt_bias"], (0.0, INF))
    ys, h = loop(
        ctx, f["step_x"], f["step_dt_raw"], f["step_B"], f["step_C"], *args, h=f["step_h0"]
    )
    check.allclose(ys, f["step_y"], **TOL, label="y of the six tokens")
    check.allclose(h, f["step_h"], **TOL, label="h after the last token")


# Rejects decay_includes_s, matrix_dt_of_target, and read_before_update, whose step
# and matrix no longer describe the same layer.
@case("M X + D X equals the step loop from a zero state, 9 tokens", public=True)
def matrix_equals_loop(ctx):
    def prop(rng):
        a_log, d, dt_bias = params(ctx)
        x, dt_raw = ctx.randn(T, H, P), ctx.randn(T, H, scale=2.0)
        b, c = ctx.randn(T, G, N), ctx.randn(T, G, N)
        ys, _ = loop(ctx, x, dt_raw, b, c, a_log, d, dt_bias, NEMOTRON)
        dt = np.clip(softplus(dt_raw + dt_bias), 0.001, None).astype(np.float32)
        m = ctx.call(MATRIX, dt, b, c, a_log)
        check.shape(m, (H, T, T), label="M")
        via_matrix = np.einsum("hts,shp->thp", m, x) + d[None, :, None] * x
        check.allclose(via_matrix, ys, **TOL, label="M X + D X against the step loop")

    check.forall(prop, trials=4)


# Rejects state_in_place: the caller keeps the state it passed in.
@case("the tensors passed in, the state included, are not modified", public=True)
def inputs_unchanged(ctx):
    a_log, d, dt_bias = params(ctx)
    arrays = (ctx.randn(H, P, N), ctx.randn(H, P), ctx.randn(H), ctx.randn(G, N), ctx.randn(G, N),
              a_log, d, dt_bias)  # fmt: skip
    tensors = [ctx.fw.tensor(a) for a in arrays]
    ctx.solution.ssd_step(*tensors, NEMOTRON)
    names = ("h", "x", "dt_raw", "B", "C", "A_log", "D", "dt_bias")
    for name, tensor, before in zip(names, tensors, arrays, strict=True):
        check.allclose(ctx.fw.numpy(tensor), before, rtol=0, atol=0, label=f"{name} after the call")


@case("the matrix against the reference, and zero above the diagonal")
def matrix_values(ctx):
    a_log, _d, _bias = params(ctx)
    dt = ctx.uniform(T, H, low=0.001, high=2.0)
    m = ctx.check(MATRIX, dt, ctx.randn(T, G, N), ctx.randn(T, G, N), a_log, **TOL)
    upper = np.triu(np.ones((T, T), bool), 1)
    check.that(
        bool((m[:, upper] == 0).all()),
        "M: an entry above the diagonal is not zero (an output would depend on a later input)",
    )


# Rejects clamp_lower_only: softplus(120) = 120 is cut to 100, the upper limit that
# mlx-lm's ssm_update has as its default.
@case("a step size above dt_limit[1]: limits (0.001, 100) and a raw value of 120")
def upper_clamp(ctx):
    a_log, d, dt_bias = params(ctx)
    dt_raw = f32(120.0, 1.0, 150.0, -2.0) - dt_bias
    step(ctx, ctx.randn(H, P, N), ctx.randn(H, P), dt_raw,
         ctx.randn(G, N), ctx.randn(G, N), a_log, d, dt_bias, (0.001, 100.0))  # fmt: skip


# Rejects hardcoded_limit: with a lower limit of 0 nothing is clamped, and a step size
# of 4.5e-5 must stay 4.5e-5.
@case("limits (0, inf): a tiny step size is kept, the state barely moves")
def no_limit(ctx):
    a_log, d, dt_bias = params(ctx)
    dt_raw = f32(-10.0, 0.5, -10.0, 2.0) - dt_bias
    h, x = ctx.randn(H, P, N), 50.0 * ctx.randn(H, P)
    b, c = ctx.randn(G, N), ctx.randn(G, N)
    _, new = step(ctx, h, x, dt_raw, b, c, a_log, d, dt_bias, (0.0, INF))
    # The two quiet heads move by dt * x * B, about 4.5e-5 * 50 * B. Compare that change
    # on its own scale: a clamp to 0.001 would make it 22 times larger.
    want = ctx.expected(STEP, h, x, dt_raw, b, c, a_log, d, dt_bias, (0.0, INF))[1]
    quiet = [0, 2]
    check.allclose(
        (new - h)[quiet], (want - h)[quiet], rtol=2e-2, atol=1e-4, label="h minus the h passed in"
    )


# The library's multi-token scan with Nemotron's clamp, several step sizes below it.
@case("nine tokens with the clamp agree with the chunked scan of Hugging Face Transformers")
def shipped_scan(ctx):
    f = ctx.fixture("shipped.npz")
    x, dt_raw, b, c = (f[f"scan_{name}"] for name in ("x", "dt_raw", "B", "C"))
    ys, h = loop(ctx, x, dt_raw, b, c, f["A_log"], f["D"], f["dt_bias"], NEMOTRON)
    check.allclose(ys, f["scan_y"], **TOL, label="y")
    check.allclose(h, f["scan_h"], **TOL, label="h")
    dt = np.clip(softplus(dt_raw + f["dt_bias"]), 0.001, None).astype(np.float32)
    m = ctx.call(MATRIX, dt, b, c, f["A_log"])
    via_matrix = np.einsum("hts,shp->thp", m, x) + f["D"][None, :, None] * x
    check.allclose(via_matrix, f["scan_y"], **TOL, label="M X + D X")


# Rejects groups_tiled where it is not hidden: 4 heads in 2 groups is (0, 0, 1, 1).
# With one group, or one group per head, repeat and tile coincide.
@case("one group for all heads, and one group per head")
def group_layouts(ctx):
    a_log, d, dt_bias = params(ctx)
    for groups in (1, H):
        step(ctx, ctx.randn(H, P, N), ctx.randn(H, P), ctx.randn(H),
             ctx.randn(groups, N), ctx.randn(groups, N), a_log, d, dt_bias, NEMOTRON)  # fmt: skip
        ctx.check(MATRIX, ctx.uniform(T, H, low=0.01, high=1.5), ctx.randn(T, groups, N),
                  ctx.randn(T, groups, N), a_log, **TOL)  # fmt: skip


# Square everything: a factor broadcast along the wrong axis still has a legal shape.
@case("heads, head width, and state size all 4")
def square_shapes(ctx):
    a_log, d, dt_bias = params(ctx)
    step(ctx, ctx.randn(4, 4, 4), ctx.randn(4, 4), ctx.randn(4, scale=2.0),
         ctx.randn(2, 4), ctx.randn(2, 4), a_log, d, dt_bias, NEMOTRON)  # fmt: skip
    ctx.check(MATRIX, ctx.uniform(4, 4, low=0.01, high=1.5), ctx.randn(4, 2, 4),
              ctx.randn(4, 2, 4), a_log, **TOL)  # fmt: skip


@case("a sequence of one token: the matrix is the single number (C . B) * dt per head")
def one_token_matrix(ctx):
    a_log, _d, _bias = params(ctx)
    dt, b, c = ctx.uniform(1, H, low=0.01, high=1.5), ctx.randn(1, G, N), ctx.randn(1, G, N)
    m = ctx.check(MATRIX, dt, b, c, a_log, **TOL)
    want = np.repeat(np.sum(c[0] * b[0], -1), H // G) * dt[0]
    check.allclose(m.reshape(-1), want, **TOL, label="M")


# The state is everything: nine tokens from a zero state equal four tokens, then five
# more from the state after the fourth, read through the matrix of all nine.
@case("the state after 4 tokens carries the sequence: rows 4 to 8 of M X + D X")
def continued(ctx):
    a_log, d, dt_bias = params(ctx)
    x, dt_raw = ctx.randn(T, H, P), ctx.randn(T, H, scale=2.0)
    b, c = ctx.randn(T, G, N), ctx.randn(T, G, N)
    _, mid = loop(ctx, x[:4], dt_raw[:4], b[:4], c[:4], a_log, d, dt_bias, NEMOTRON)
    ys, _ = loop(ctx, x[4:], dt_raw[4:], b[4:], c[4:], a_log, d, dt_bias, NEMOTRON, h=mid)
    dt = np.clip(softplus(dt_raw + dt_bias), 0.001, None).astype(np.float32)
    m = ctx.expected(MATRIX, dt, b, c, a_log)
    want = np.einsum("hts,shp->thp", m, x) + d[None, :, None] * x
    check.allclose(ys, want[4:], **TOL, label="y")
