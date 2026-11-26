"""Tests for gptq-column-update.

`gptq_quantize` returns integer codes, compared exactly. A code is a rounding of a
weight that has absorbed the corrections of all earlier columns, so a float32 result
may differ from an exact one by about 1e-4 of a step, and a value near a rounding tie
could go either way. `tie_free` therefore replays the algorithm in float64 and nudges
every weight that reaches its own column within MARGIN of a tie (the nudge changes
nothing before that column, since a column's corrections do not depend on its own
value). The factor handed to `gptq_quantize` is computed here in float64 and passed
as float32, the same numbers for every framework.

`inverse_hessian_factor` is compared with that float64 factor. A float32 Cholesky of
the inverse of a matrix with condition number near 1000 (what 1 percent dampening
allows) is accurate to about 5e-6 of the largest entry in all three frameworks; the
tolerance is 1e-3 relative plus 1e-4 of the largest entry.

JAX compiles per shape: the cases share (8, 16), (8, 32), and (4, 8).
"""

import numpy as np

from learnai.judge import case, check

MARGIN = 0.02


def hessian(x, percdamp=0.0):
    x = x.astype(np.float64)
    h = (2.0 / x.shape[0]) * (x.T @ x)
    return h + percdamp * np.mean(np.diag(h)) * np.eye(x.shape[1])


def factor(x, percdamp):
    """The upper Cholesky factor of the dampened inverse Hessian, in float64."""
    return np.linalg.cholesky(np.linalg.inv(hessian(x, percdamp))).T


def row_scale(w, bits):
    qmax = 2 ** (bits - 1) - 1
    max_abs = np.abs(w.astype(np.float64)).max(axis=1)
    return np.where(max_abs > 0, max_abs / qmax, 1.0), qmax


def replay(w, u, bits, *, fix=False):
    """The algorithm in float64. Returns (codes, w, clean): with fix=True, w is a copy in
    which weights near a tie were nudged; clean says whether no tie was left."""
    w0 = np.array(w, dtype=np.float32)
    scale, qmax = row_scale(w0, bits)
    work = w0.astype(np.float64)
    u = u.astype(np.float64)
    codes, clean = [], True
    for j in range(work.shape[1]):
        t = work[:, j] / scale
        near = (np.abs(t - np.floor(t) - 0.5) < MARGIN) & (np.abs(t) < qmax + 0.5)
        if near.any():
            if fix:
                # Move toward zero by a tenth of a step: never a new row maximum.
                delta = np.where(near, -np.sign(w0[:, j]) * 0.1 * scale, 0.0)
                delta = np.where(near & (w0[:, j] == 0), 0.1 * scale, delta)
                before = w0[:, j].copy()
                w0[:, j] = (w0[:, j] + delta).astype(np.float32)
                work[:, j] += w0[:, j].astype(np.float64) - before
                t = work[:, j] / scale
            else:
                clean = False
        q = np.clip(np.round(t), -qmax, qmax)
        codes.append(q)
        err = (work[:, j] - q * scale) / u[j, j]
        work = work - np.outer(err, u[j])
    return np.stack(codes, axis=1).astype(np.int64), w0, clean


def tie_free(ctx, make_w, u, bits):
    """Weights for which no rounding of the replay is within MARGIN of a tie."""
    for _ in range(100):
        w = np.asarray(make_w(), dtype=np.float32)
        _, fixed, _ = replay(w, u, bits, fix=True)
        same_scale = np.array_equal(np.abs(fixed).max(axis=1), np.abs(w).max(axis=1))
        if same_scale and replay(fixed, u, bits)[2]:
            return fixed
    raise RuntimeError("could not make the input tie-free")


def calibration(ctx, n, in_dim, *, mixing=1.0):
    """Correlated calibration inputs whose channels differ in magnitude."""
    base = ctx.rng.standard_normal((n, in_dim))
    mix = np.eye(in_dim) + mixing * ctx.rng.standard_normal((in_dim, in_dim)) / np.sqrt(in_dim)
    magnitude = 10.0 ** ctx.rng.uniform(-0.5, 0.5, in_dim)
    return ((base @ mix) * magnitude).astype(np.float32)


def output_error(w, w_hat, x):
    """Squared error of the layer's output on the calibration inputs, in float64."""
    diff = (w.astype(np.float64) - w_hat) @ x.astype(np.float64).T
    return float((diff**2).sum())


def round_to_nearest(w, bits):
    scale, qmax = row_scale(w, bits)
    return np.clip(np.round(w.astype(np.float64) / scale[:, None]), -qmax, qmax) * scale[:, None]


def check_factor(ctx, x, percdamp):
    x = np.asarray(x, dtype=np.float32)
    want = factor(x, percdamp)
    got = ctx.call("inverse_hessian_factor", x, percdamp)
    oracle = ctx.expected("inverse_hessian_factor", x, percdamp)
    atol = 1e-4 * float(np.abs(want).max())
    check.allclose(oracle, want, rtol=1e-4, atol=atol / 10, label="oracle")
    check.shape(got, want.shape, label="inverse_hessian_factor")
    check.that(
        got.dtype == np.float32,
        f"inverse_hessian_factor: got dtype {got.dtype}, expected dtype float32",
        "expected float32",
    )
    check.finite(got, label="inverse_hessian_factor")
    check.allclose(got, want, rtol=1e-3, atol=atol, label="inverse_hessian_factor")
    return got


def check_quantize(ctx, w, u, bits):
    """gptq_quantize against the float64 replay and the oracle. w must be tie-free."""
    u32 = np.asarray(u, dtype=np.float32)
    want, _, clean = replay(w, u32, bits)
    if not clean:  # an authoring error
        raise RuntimeError("bad test input: a rounding tie")
    result = ctx.call("gptq_quantize", w, u32, bits)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        "gptq_quantize: expected a (codes, scale) pair",
    )
    codes, scale = result
    oracle_codes, oracle_scale = ctx.expected("gptq_quantize", w, u32, bits)
    if not np.array_equal(oracle_codes, want):  # an authoring error
        raise RuntimeError("the oracle disagrees with the float64 replay")
    check.shape(scale, (w.shape[0],), label="scale")
    check.allclose(scale, oracle_scale, label="scale", dtype=True)
    check.shape(codes, w.shape, label="codes")
    check.that(codes.dtype == np.int8, "codes must be an int8 tensor", f"dtype {codes.dtype}")
    check.equal(codes, want, label="codes")
    return codes, scale


W_EXAMPLE = np.array([[0.7, 1.5, -0.8], [1.4, -3.0, 0.4]], dtype=np.float32)
U_EXAMPLE = np.array([[1.0, 0.5, -0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


@case("the worked example: a (2, 3) weight, 3 bits, a given factor", public=True)
def worked_example(ctx):
    codes, scale = ctx.call("gptq_quantize", W_EXAMPLE, U_EXAMPLE, 3)
    check.allclose(scale, np.array([0.5, 1.0], np.float32), label="scale")
    check.equal(codes, np.array([[1, 3, -1], [1, -3, 1]]), label="codes")
    check_quantize(ctx, W_EXAMPLE, U_EXAMPLE, 3)


# Rejects no_dampening, absolute_dampening, lower_factor, factor_of_hessian,
# unfactored_inverse, hessian_not_averaged.
@case("inverse_hessian_factor: two samples of two channels, then 64 samples of 16", public=True)
def factor_basic(ctx):
    x = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    got = check_factor(ctx, x, 0.0)  # H = [[3, 0], [0, 3]]: U = I / sqrt(3)
    check.allclose(got, np.eye(2) / np.sqrt(3.0), atol=1e-5, label="inverse_hessian_factor")
    x = np.array([[2.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    check_factor(ctx, x, 0.5)
    x = calibration(ctx, 64, 16)
    got = check_factor(ctx, x, 0.01)
    check.that(
        float(np.abs(np.tril(got, -1)).max()) <= 1e-6 * float(np.abs(got).max()),
        "inverse_hessian_factor: the result is not upper triangular",
    )
    check_factor(ctx, x * 30.0, 0.01)
    check_factor(ctx, x * 0.02, 0.1)


# Rejects plain_rounding, no_division, first_diagonal, squared_diagonal, adds_error,
# updates_with_column, inverse_hessian_rows.
@case("gptq_quantize: an (8, 16) layer at 4 bits with correlated inputs", public=True)
def quantize_basic(ctx):
    x = calibration(ctx, 64, 16)
    u = factor(x, 0.01)
    for _ in range(2):
        w = tie_free(ctx, lambda: ctx.randn(8, 16, scale=0.1), u, 4)
        check_quantize(ctx, w, u, 4)


@case("the result beats round-to-nearest on the calibration objective", public=True)
def beats_round_to_nearest(ctx):
    x = calibration(ctx, 64, 16, mixing=2.0)
    u = factor(x, 0.01)
    w = tie_free(ctx, lambda: ctx.randn(8, 16, scale=0.1), u, 3)
    codes, scale = check_quantize(ctx, w, u, 3)
    ours = output_error(w, codes * scale.astype(np.float64)[:, None], x)
    plain = output_error(w, round_to_nearest(w, 3), x)
    check.at_most("error_vs_round_to_nearest", ours / plain, 0.9)


# Rejects no_clamp and scale_from_updated_weights.
@case("corrections that push a weight past the row's range: codes stay on the grid", public=True)
def clamps_codes(ctx):
    x = calibration(ctx, 64, 16, mixing=3.0)
    u = factor(x, 0.01)
    for _ in range(200):
        w = tie_free(ctx, lambda: ctx.uniform(8, 16, low=-0.1, high=0.1), u, 2)
        scale, qmax = row_scale(w, 2)
        unclamped, work = False, w.astype(np.float64)
        for j in range(16):
            t = work[:, j] / scale
            unclamped |= bool((np.abs(np.round(t)) > qmax).any())
            q = np.clip(np.round(t), -qmax, qmax)
            work = work - np.outer((work[:, j] - q * scale) / u[j, j], u[j])
        if unclamped:
            break
    else:  # an authoring error
        raise RuntimeError("no input needs the clamp")
    codes, _ = check_quantize(ctx, w, u, 2)
    check.that(int(np.abs(codes).max()) <= 1, "codes: a code outside [-qmax, qmax]")


@case("all bit widths; an all-zero row; a row with a single non-zero weight")
def bit_widths_and_zero_rows(ctx):
    x = calibration(ctx, 64, 16)
    u = factor(x, 0.01)

    def make():
        w = ctx.randn(8, 16, scale=0.3)
        w[2] = 0.0
        w[5] = 0.0
        w[5, 7] = -0.25
        return w

    for bits in (2, 3, 5, 6, 8):
        w = tie_free(ctx, make, u, bits)
        codes, scale = check_quantize(ctx, w, u, bits)
        check.that(
            not codes[2].any() and float(scale[2]) == 1.0,
            "an all-zero row must get codes 0 and scale 1",
        )


@case("inverse_hessian_factor: fewer samples than channels needs the dampening")
def factor_rank_deficient(ctx):
    x = calibration(ctx, 12, 32)
    got = check_factor(ctx, x, 0.01)
    check_factor(ctx, x, 0.1)
    h = hessian(x, 0.01)
    residual = got.astype(np.float64).T @ got.astype(np.float64) @ h - np.eye(32)
    check.that(
        float(np.abs(residual).max()) < 2e-2,
        "inverse_hessian_factor: U^T U is not the inverse of the dampened Hessian",
        f"largest entry of U^T U H - I: {np.abs(residual).max():.3g}",
    )
    check_factor(ctx, calibration(ctx, 4, 8), 0.05)


@case("wider layers, square shapes, and strongly correlated inputs")
def other_shapes(ctx):
    x = calibration(ctx, 96, 32, mixing=3.0)
    u = factor(x, 0.01)
    check_factor(ctx, x, 0.01)
    check_quantize(ctx, tie_free(ctx, lambda: ctx.randn(8, 32, scale=0.05), u, 4), u, 4)
    x = calibration(ctx, 16, 16)
    u = factor(x, 0.01)
    check_factor(ctx, x, 0.01)
    check_quantize(ctx, tie_free(ctx, lambda: ctx.randn(16, 16), u, 4), u, 4)
    x = calibration(ctx, 4, 8)
    u = factor(x, 0.05)
    check_quantize(ctx, tie_free(ctx, lambda: ctx.randn(4, 8, scale=2.0), u, 5), u, 5)
    check_quantize(ctx, tie_free(ctx, lambda: ctx.randn(1, 8), u, 4), u, 4)


@case("end to end: your factor, then your quantizer, beats round-to-nearest")
def end_to_end(ctx):
    def prop(rng):
        n, in_dim = ((64, 16), (96, 32), (12, 32))[int(rng.integers(3))]
        bits = int(rng.integers(3, 6))
        x = calibration(ctx, n, in_dim, mixing=float(rng.uniform(1.0, 3.0)))
        w = ctx.randn(8, in_dim, scale=float(10.0 ** rng.uniform(-2, 0)))
        u = check_factor(ctx, x, 0.01)
        codes, scale = ctx.call("gptq_quantize", w, u, bits)
        check.shape(codes, w.shape, label="codes")
        qmax = 2 ** (bits - 1) - 1
        check.that(
            codes.dtype.kind == "i" and int(np.abs(codes).max()) <= qmax,
            "codes must be integers in [-qmax, qmax]",
        )
        check.allclose(scale, row_scale(w, bits)[0], label="scale")
        ours = output_error(w, codes * scale.astype(np.float64)[:, None], x)
        plain = output_error(w, round_to_nearest(w, bits), x)
        check.that(
            ours <= 0.9 * plain,
            "the output error is not at least 10 percent below round-to-nearest",
            f"bits = {bits}, shape {w.shape}: error {ours:.6g}, round-to-nearest {plain:.6g}",
        )

    check.forall(prop, trials=6)
