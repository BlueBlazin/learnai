"""Tests for awq-scale-search.

The answer is one grid entry and its scales, so a case is only fair if float32
rounding cannot change which entry wins. `losses` evaluates the statement in float64,
and `separated` drops every grid entry whose loss is within 2 percent of the best one
(other than the best itself). Float32 and float64 losses agree to about 1e-5 relative
on these inputs, so the margin is far above the noise. The tie case uses activations
of magnitude exactly 1, for which every scale is exactly 1 and all losses are
bit-identical.

JAX compiles per shape: the cases share w (8, 32) with x (64, 32), and a few small
layouts.
"""

import numpy as np

from learnai.judge import case, check

GAP = 1.02
AWQ_GRID = (np.arange(20) / 20).astype(np.float32)  # 0, 0.05, ..., 0.95


def quantize_groups(w, bits, group_size):
    """Group-wise quantize and dequantize, as in groupwise-4bit, in float64."""
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.min(axis=-1, keepdims=True)
    hi = groups.max(axis=-1, keepdims=True)
    scale = np.where(hi > lo, (hi - lo) / levels, 1.0)
    codes = np.clip(np.round((groups - lo) / scale), 0, levels)
    return (codes * scale + lo).reshape(out_dim, in_dim)


def scales_for(x, alpha):
    act = np.abs(x.astype(np.float64)).mean(axis=0)
    s = np.maximum(act ** np.float64(alpha), 1e-4)
    return s / np.sqrt(s.max() * s.min())


def output_loss(w, x, bits, group_size, s):
    """Mean squared output error of the layer quantized under the scales s, in float64."""
    w, x = w.astype(np.float64), x.astype(np.float64)
    w_hat = quantize_groups(w * s, bits, group_size) / s
    return float(((x @ w.T - x @ w_hat.T) ** 2).mean())


def losses(w, x, bits, group_size, grid):
    return np.array([output_loss(w, x, bits, group_size, scales_for(x, a)) for a in grid])


def separated(w, x, bits, group_size, grid):
    """The grid without the runners-up within 2 percent of the best loss; order is kept."""
    grid = np.asarray(grid, dtype=np.float32)
    loss = losses(w, x, bits, group_size, grid)
    keep = loss >= loss.min() * GAP
    keep[np.argmin(loss)] = True
    return grid[keep]


def check_awq(ctx, w, x, bits, group_size, grid, *, exact_ties=False):
    """Compare with the float64 model and the oracle; returns (scales, alpha, grid used)."""
    w, x = np.asarray(w, dtype=np.float32), np.asarray(x, dtype=np.float32)
    grid = np.asarray(grid, dtype=np.float32)
    if not exact_ties:
        grid = separated(w, x, bits, group_size, grid)
        want_alpha = grid[np.argmin(losses(w, x, bits, group_size, grid))]
    else:
        want_alpha = grid[0]
    result = ctx.call("awq_scales", w, x, bits, group_size, grid)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        "awq_scales: expected a (scales, alpha) pair",
    )
    scales, alpha = result
    oracle_scales, oracle_alpha = ctx.expected("awq_scales", w, x, bits, group_size, grid)
    if float(oracle_alpha) != float(want_alpha):  # an authoring error
        raise RuntimeError(f"the oracle chose {oracle_alpha}, the float64 model {want_alpha}")
    check.shape(alpha, (), label="alpha")
    check.that(
        float(alpha) == float(want_alpha),
        f"alpha: returned {float(alpha):.4g}, expected {float(want_alpha):.4g} (the first grid "
        "entry with the lowest output error)",
        f"grid = {grid.tolist()}\n"
        f"output error of each entry = {losses(w, x, bits, group_size, grid).tolist()}\n"
        f"expected alpha = {float(want_alpha):.4g}",
    )
    check.shape(scales, (w.shape[1],), label="scales")
    check.allclose(scales, oracle_scales, label="scales", dtype=True)
    check.allclose(scales, scales_for(x, want_alpha), label="scales")
    return scales, float(alpha), grid


def layer(ctx, out_dim=8, in_dim=32, n=64, *, spread=1.0, weight_scale=0.1):
    """A weight matrix and calibration inputs whose channels differ in magnitude."""
    w = ctx.randn(out_dim, in_dim, scale=weight_scale)
    magnitude = 10.0 ** ctx.uniform(in_dim, low=-spread, high=spread)
    x = ctx.randn(n, in_dim) * magnitude
    return w, x.astype(np.float32)


W_EXAMPLE = np.array([[-0.2, 0.5, 0.7, 1.0]], dtype=np.float32)
X_EXAMPLE = np.array([[1.0, 1.0, 16.0, 1.0], [-1.0, -1.0, 16.0, 1.0]], dtype=np.float32)


@case("the worked example: one row of four weights, 2 bits, grid 0, 0.5, 1", public=True)
def worked_example(ctx):
    grid = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    table = losses(W_EXAMPLE, X_EXAMPLE, 2, 4, grid)
    if not np.allclose(table, [2.57, 0.13, 1.93], rtol=1e-6):  # an authoring error
        raise RuntimeError(f"the losses of the statement are wrong: {table}")
    scales, alpha = ctx.call("awq_scales", W_EXAMPLE, X_EXAMPLE, 2, 4, grid)
    check.allclose(alpha, np.float32(0.5), label="alpha")
    check.allclose(scales, np.array([0.5, 0.5, 2.0, 0.5], np.float32), label="scales")
    check_awq(ctx, W_EXAMPLE, X_EXAMPLE, 2, 4, grid)


# Rejects scales_weights_only, weight_error, quantizes_before_scaling, returns_last_scales.
@case("an (8, 32) layer whose input channels span two decades, the AWQ grid", public=True)
def channels_of_different_magnitude(ctx):
    for _ in range(2):
        w, x = layer(ctx)
        _, alpha, _ = check_awq(ctx, w, x, 4, 8, AWQ_GRID)
        check.that(alpha > 0, "alpha: with channels this uneven, some scaling must beat none")


# Rejects signed_mean (NaN for a negative mean) and rms_activation.
@case("signed, heavy-tailed activations: the magnitude is the mean of |x|", public=True)
def signed_heavy_tailed(ctx):
    w = ctx.randn(8, 32, scale=0.1)
    magnitude = 10.0 ** ctx.uniform(32, low=-1.0, high=1.0)
    x = (ctx.rng.standard_t(2.5, size=(64, 32)) * magnitude).astype(np.float32)
    x[:, ::2] -= np.abs(x[:, ::2]).mean(axis=0) * 0.5  # some channels have a negative mean
    check_awq(ctx, w, x, 4, 8, AWQ_GRID)
    check_awq(ctx, w, x, 3, 16, AWQ_GRID)


# Rejects ties_to_last.
@case("equal channel magnitudes: every alpha ties, and the first grid entry wins", public=True)
def exact_tie(ctx):
    w = ctx.randn(8, 32, scale=0.1)
    x = np.where(ctx.rng.random((64, 32)) < 0.5, -1.0, 1.0).astype(np.float32)
    for grid in ([0.5, 0.0, 1.0, 0.25], [0.75, 0.25], [0.0, 1.0]):
        scales, _, _ = check_awq(ctx, w, x, 4, 8, grid, exact_ties=True)
        check.that(bool((scales == 1.0).all()), "scales: equal channels must all get scale 1")


# Rejects no_clamp.
@case("dead channels: an input that is always zero gets the clamped scale", public=True)
def dead_channels(ctx):
    w, x = layer(ctx)
    x[:, [3, 17]] = 0.0
    scales, alpha, _ = check_awq(ctx, w, x, 4, 8, AWQ_GRID)
    check.finite(scales, label="scales")
    check.that(alpha > 0, "alpha: a dead channel must not force alpha = 0")


# Rejects no_normalization and max_normalization.
@case("the returned scales are normalized: max(s) * min(s) = 1")
def normalization(ctx):
    w, x = layer(ctx, spread=1.5)
    for grid in (AWQ_GRID[4:], [1.0, 0.5], [0.3]):
        scales, _, _ = check_awq(ctx, w, x, 4, 8, grid)
        product = float(scales.max()) * float(scales.min())
        check.that(
            abs(product - 1.0) < 1e-3,
            "scales: the largest and the smallest scale must be reciprocal",
            f"max(s) * min(s) = {product:.6g}",
        )


@case("other bit widths, group sizes, and shapes; grids in any order")
def other_configurations(ctx):
    w, x = layer(ctx)
    for bits, group_size in ((2, 8), (3, 32), (6, 4), (8, 8), (4, 1)):
        check_awq(ctx, w, x, bits, group_size, AWQ_GRID)
    order = ctx.rng.permutation(20)
    check_awq(ctx, w, x, 4, 8, AWQ_GRID[order])
    check_awq(ctx, w, x, 4, 8, np.linspace(1.0, 0.0, 11, dtype=np.float32))
    w, x = layer(ctx, 1, 8, 4)  # a single output row, fewer samples than channels
    check_awq(ctx, w, x, 4, 4, AWQ_GRID)
    w, x = layer(ctx, 16, 16, 16)  # square everything
    check_awq(ctx, w, x, 4, 8, AWQ_GRID)


# Rejects sum_of_abs_error.
@case("a few large calibration samples: the objective is the squared error")
def squared_error(ctx):
    found = False
    for _ in range(40):
        w, x = layer(ctx)
        x[:4] *= 12.0  # a handful of samples dominate a squared error, not an absolute one
        grid = separated(w, x, 4, 8, AWQ_GRID)
        s = [scales_for(x, a) for a in grid]
        w64, x64 = w.astype(np.float64), x.astype(np.float64)
        absolute = [
            np.abs(x64 @ w64.T - x64 @ (quantize_groups(w64 * v, 4, 8) / v).T).mean() for v in s
        ]
        if np.argmin(absolute) != np.argmin(losses(w, x, 4, 8, grid)):
            found = True
            check_awq(ctx, w, x, 4, 8, grid)
            break
    if not found:  # an authoring error
        raise RuntimeError("no input separates the squared from the absolute error")


@case("the chosen scales never do worse on the calibration set than no scaling")
def not_worse_than_unscaled(ctx):
    def prop(rng):
        spread = float(rng.uniform(0.2, 1.5))
        w, x = layer(ctx, spread=spread, weight_scale=float(10.0 ** rng.uniform(-2, 0)))
        bits = int(rng.integers(3, 6))
        scales, alpha, grid = check_awq(ctx, w, x, bits, 8, AWQ_GRID)
        chosen = output_loss(w, x, bits, 8, scales.astype(np.float64))
        unscaled = output_loss(w, x, bits, 8, np.ones(w.shape[1]))
        check.that(
            chosen <= unscaled * (1 + 1e-4),
            "the returned scales give a larger output error than no scaling at all",
            f"alpha = {alpha}: output error {chosen:.6g}, unscaled {unscaled:.6g}",
        )
        best = losses(w, x, bits, 8, grid).min()
        check.that(
            chosen <= best * (1 + 1e-3),
            "the returned scales do not reach the lowest output error of the grid",
            f"output error {chosen:.6g}, best of the grid {best:.6g}",
        )

    check.forall(prop, trials=6)
