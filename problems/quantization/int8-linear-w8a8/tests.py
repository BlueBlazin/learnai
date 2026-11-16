"""Tests for int8-linear-w8a8.

The accumulator is compared exactly, so no activation may sit near a rounding tie:
frameworks could then round x / scale to different sides because of a one-ulp
difference in the scale. Every row of activations goes through `tie_free_row`, which
keeps x / scale and lo / scale at least 0.05 away from a half-integer, as the statement
promises. With that margin the codes can also be computed here, in numpy, so `acc` is
checked against integers that do not come from any reference.

JAX compiles every new shape, so the cases share a few layouts.
"""

import numpy as np

from learnai.judge import case, check

QMIN, QMAX = -128, 127
MARGIN = 0.05


def near_tie(v):
    """True where v (float64) is within MARGIN of a half-integer."""
    return np.abs(v - np.floor(v) - 0.5) < MARGIN


def grid(row):
    """(lo, hi, scale) of one token, in float32 like a solution computes them."""
    lo, hi = np.float32(min(row.min(), 0.0)), np.float32(max(row.max(), 0.0))
    scale = (hi - lo) / np.float32(QMAX - QMIN) if hi > lo else np.float32(1.0)
    return lo, hi, np.float32(scale)


def tie_free_row(make):
    """Draw a row until its zero-point is clear of a tie, then move every element whose
    code would be near a tie onto the nearest grid point of the row."""
    for _ in range(200):
        row = np.asarray(make(), dtype=np.float32)
        lo, hi, scale = grid(row)
        if near_tie(np.float64(lo) / np.float64(scale)):
            continue
        t = row.astype(np.float64) / np.float64(scale)
        bad = near_tie(t)
        row[bad] = np.clip(np.round(t[bad]) * np.float64(scale), lo, hi).astype(np.float32)
        t = row.astype(np.float64) / np.float64(scale)
        if grid(row) == (lo, hi, scale) and not near_tie(t).any():
            return row
    raise AssertionError("could not draw a tie-free row")


def tokens(ctx, n, in_dim, *, magnitudes=None, shift=0.0):
    """n tie-free rows of normal activations; row r has spread magnitudes[r]."""
    magnitudes = np.ones(n) if magnitudes is None else magnitudes
    return np.stack(
        [
            tie_free_row(lambda m=m: ctx.randn(in_dim, scale=float(m)) + np.float32(shift * m))
            for m in magnitudes
        ]
    )


def activation_codes(x):
    """(codes, scale, zero_point) per row, as the statement defines them."""
    codes = np.empty(x.shape, dtype=np.int64)
    scales = np.empty(len(x), dtype=np.float64)
    zero_points = np.empty(len(x), dtype=np.int64)
    for r, row in enumerate(x):
        lo, _, scale = grid(row)
        z = int(np.clip(QMIN - np.round(np.float64(lo) / np.float64(scale)), QMIN, QMAX))
        q = np.round(row.astype(np.float64) / np.float64(scale)) + z
        codes[r], scales[r], zero_points[r] = np.clip(q, QMIN, QMAX), scale, z
    return codes, scales, zero_points


def layer(ctx, in_dim, out_dim, *, bias_scale=1.0):
    """Random int8 weights in [-127, 127] with per-row scales and a float bias."""
    w_q = ctx.randint(-127, 128, (out_dim, in_dim)).astype(np.int8)
    w_scale = (10.0 ** ctx.uniform(out_dim, low=-3.5, high=-2.0)).astype(np.float32)
    return w_q, w_scale, ctx.randn(out_dim, scale=bias_scale)


def check_layer(ctx, x, w_q, w_scale, bias):
    """Compare with the oracle, with integers computed here, and with the float layer."""
    n, out_dim = x.shape[0], w_q.shape[0]
    y, acc = ctx.call("int8_linear", x, w_q, w_scale, bias)
    want_y, want_acc = ctx.expected("int8_linear", x, w_q, w_scale, bias)

    check.shape(acc, (n, out_dim), label="acc")
    check.that(acc.dtype == np.int32, "acc must be an int32 tensor", f"got dtype {acc.dtype}")
    codes, scales, _ = activation_codes(x)
    exact = codes @ w_q.astype(np.int64).T
    assert np.array_equal(exact, want_acc), "the reference accumulator is not exact"
    check.equal(acc, exact, label="acc")

    check.shape(y, (n, out_dim), label="y")
    check.allclose(y, want_y, atol=1e-5, dtype=True, label="y")

    # The quantization bound: half an activation step times the L1 norm of the weight row.
    w_hat = w_q.astype(np.float64) * w_scale.astype(np.float64)[:, None]
    float_layer = x.astype(np.float64) @ w_hat.T + bias.astype(np.float64)
    bound = 0.5 * scales[:, None] * np.abs(w_hat).sum(axis=1)[None, :]
    slack = 1e-3 * bound + 1e-4 * np.abs(float_layer) + 1e-5
    error = np.abs(y.astype(np.float64) - float_layer)
    worst = np.unravel_index(np.argmax(error - bound - slack), error.shape)
    check.that(
        bool((error <= bound + slack).all()),
        "y is further from the float layer x @ W_hat.T + bias than quantizing x allows",
        f"token {int(worst[0])}, output {int(worst[1])}: |error| = {error[worst]:.6g}, "
        f"bound = {bound[worst]:.6g}",
    )
    return y, acc


@case("the worked example: two tokens, five inputs, two outputs", public=True)
def worked_example(ctx):
    x = np.array([[-0.5, 0.0, 0.3, 1.2, 2.0], [2.0, 3.0, 5.0, 0.0, 1.0]], dtype=np.float32)
    w_q = np.array([[127, -64, 0, 32, -127], [10, 20, 30, 40, 50]], dtype=np.int8)
    w_scale = np.array([0.01, 0.002], dtype=np.float32)
    bias = np.array([0.5, -1.0], dtype=np.float32)
    y, acc = ctx.call("int8_linear", x, w_q, w_scale, bias)
    check.equal(acc, np.array([[-26017, 3950], [781, -4920]]), label="acc")
    check.allclose(
        y, np.array([[-2.292255, -0.696078], [-0.15, -0.44]], np.float32), atol=1e-5, label="y"
    )
    check_layer(ctx, x, w_q, w_scale, bias)


# Rejects int8_accumulator, bias_before_rescale, correction_per_input, returns_corrected.
@case("a random batch: 4 tokens, 16 inputs, 3 outputs", public=True)
def random_batch(ctx):
    check_layer(ctx, tokens(ctx, 4, 16), *layer(ctx, 16, 3))


# Rejects per_batch_scale.
@case("tokens on very different scales each get their own grid", public=True)
def token_scales(ctx):
    x = tokens(ctx, 4, 16, magnitudes=[0.01, 1.0, 30.0, 0.3])
    check_layer(ctx, x, *layer(ctx, 16, 3))


# Rejects no_zero_point_correction and zero_point_in_int8: the zero-points are -128 and 127.
@case("non-negative tokens (zero-point -128) and non-positive tokens (zero-point 127)", public=True)
def one_sided_tokens(ctx):
    x = np.stack(
        [
            tie_free_row(lambda: np.abs(ctx.randn(16))),
            tie_free_row(lambda: np.abs(ctx.randn(16)) + 2.0),
            tie_free_row(lambda: -np.abs(ctx.randn(16))),
            tie_free_row(lambda: np.maximum(ctx.randn(16), 0.0)),
        ]
    )
    _, _, zero_points = activation_codes(x)
    assert zero_points.tolist() == [-128, -128, 127, -128]
    w_q, w_scale, bias = layer(ctx, 16, 3)
    w_q[0] = np.abs(w_q[0])  # a row whose codes do not sum to zero
    check_layer(ctx, x, w_q, w_scale, bias)


# Rejects float32_accumulator: acc[0, 0] is odd and above 2**24, so float32 cannot hold it.
@case("a wide layer: 2048 inputs, an accumulator beyond 2**24", public=True)
def wide_layer(ctx):
    in_dim = 2048
    x = np.stack(
        [
            np.full(in_dim, 3.0, dtype=np.float32),  # every code is 127
            tie_free_row(lambda: ctx.randn(in_dim)),
        ]
    )
    w_q = ctx.randint(-127, 128, (3, in_dim)).astype(np.int8)
    w_q[0] = 127
    w_q[0, 5] = 126
    w_q[1] = -127
    w_scale = np.array([1e-5, 2e-5, 1e-3], dtype=np.float32)
    bias = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    _, acc = check_layer(ctx, x, w_q, w_scale, bias)
    check.that(
        int(acc[0, 0]) == 127 * (127 * 2047 + 126),
        "the accumulator of 2048 products of saturated codes is not exact",
        f"expected {127 * (127 * 2047 + 126)}, an odd number above 2**24; got {int(acc[0, 0])}",
    )


# Rejects token_scale_on_outputs, whose shapes only fit when n == out.
@case("as many tokens as outputs")
def square_batch(ctx):
    x = tokens(ctx, 4, 16, magnitudes=[5.0, 0.02, 1.0, 0.2], shift=0.5)
    check_layer(ctx, x, *layer(ctx, 16, 4))
    check_layer(ctx, tokens(ctx, 4, 16), *layer(ctx, 16, 4))


@case("an all-zero token, a constant token, and a single token")
def degenerate_tokens(ctx):
    x = tokens(ctx, 4, 16)
    x[1] = 0.0
    x[2] = 0.75
    w_q, w_scale, bias = layer(ctx, 16, 3)
    y, _ = check_layer(ctx, x, w_q, w_scale, bias)
    check.finite(y, label="y")
    check.allclose(y[1], bias, atol=1e-6, label="y of the all-zero token (the bias alone)")
    check_layer(ctx, tokens(ctx, 1, 16), w_q, w_scale, bias)


@case("a large bias and a zero bias")
def bias_sizes(ctx):
    w_q, w_scale, _ = layer(ctx, 16, 3)
    x = tokens(ctx, 4, 16, shift=1.0)
    check_layer(ctx, x, w_q, w_scale, np.array([50.0, -20.0, 7.5], dtype=np.float32))
    check_layer(ctx, x, w_q, w_scale, np.zeros(3, dtype=np.float32))


@case("weight rows with extreme codes: all 127, all -127, and a zero row")
def extreme_weights(ctx):
    w_q, w_scale, bias = layer(ctx, 16, 3)
    w_q[0], w_q[1], w_q[2] = 127, -127, 0
    x = tokens(ctx, 4, 16, magnitudes=[1.0, 8.0, 0.1, 2.0], shift=-0.7)
    y, acc = check_layer(ctx, x, w_q, w_scale, bias)
    check.that(not acc[:, 2].any(), "a weight row of zeros must accumulate to 0")
    check.allclose(y[:, 2], np.full(4, bias[2]), atol=1e-6, label="y of the zero weight row")


@case("random tokens and layers stay within the quantization bound of the float layer")
def bound_property(ctx):
    layouts = [(4, 16, 3), (4, 16, 4), (1, 16, 3)]

    def prop(rng):
        n, in_dim, out_dim = layouts[int(rng.integers(len(layouts)))]
        magnitudes = 10.0 ** rng.uniform(-2, 2, size=n)
        x = tokens(ctx, n, in_dim, magnitudes=magnitudes, shift=float(rng.uniform(-1.5, 1.5)))
        check_layer(ctx, x, *layer(ctx, in_dim, out_dim, bias_scale=float(magnitudes.max())))

    check.forall(prop, trials=8)
