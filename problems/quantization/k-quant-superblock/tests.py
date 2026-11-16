"""Tests for k-quant-superblock.

All five outputs of `q4k_encode` are compared exactly with `model_encode`, a float64
implementation of the statement, so no rounding in the procedure may sit near a tie.
`superblock` draws one superblock at a time and makes it tie-free:

- the two 6-bit roundings (63 * value / largest) stay 0.01 away from a tie;
- the two float16 conversions give the same float16 when the value moves by 1e-5
  relative;
- every code's (w + offset) / step stays 0.02 away from a tie (0.003 in the one case
  about the float16 units): interior weights are moved a quarter of a step toward their
  grid point, and a superblock in which that would change a block's minimum or maximum
  is drawn again.

float16 arguments of `q4k_decode` are converted by hand (`ctx.call` would widen them
to float32). JAX compiles per shape: every case uses (2, 256) or (4, 256).
"""

import numpy as np

from learnai.judge import case, check

BLOCKS, BLOCK = 8, 32


def model_fields(w):
    """Level 1 and level 2 of the statement in float64, before any rounding."""
    blocks = w.astype(np.float64).reshape(-1, BLOCKS, BLOCK)
    lo = np.minimum(blocks.min(axis=-1), 0.0)
    block_scale = (blocks.max(axis=-1) - lo) / 15.0
    block_min = -lo
    max_scale = block_scale.max(axis=-1, keepdims=True)
    max_min = block_min.max(axis=-1, keepdims=True)

    def six_bit(value, largest):
        return np.where(largest > 0, 63.0 * value / np.where(largest > 0, largest, 1.0), 0.0)

    return blocks, six_bit(block_scale, max_scale), six_bit(block_min, max_min), max_scale, max_min


def model_encode(w, *, report=False, margin=0.02, exact_units=False):
    """(codes, scales, mins, d, dmin) in float64; with report=True also the tie margins."""
    blocks, ls_raw, lm_raw, max_scale, max_min = model_fields(w)
    scales, mins = np.round(ls_raw), np.round(lm_raw)
    d = (max_scale / 63.0).astype(np.float16)
    dmin = (max_min / 63.0).astype(np.float16)
    step = d.astype(np.float64) * scales
    offset = dmin.astype(np.float64) * mins
    if exact_units:  # the mistake of using the units before their float16 rounding
        step, offset = max_scale / 63.0 * scales, max_min / 63.0 * mins
    t = (blocks + offset[..., None]) / np.where(step > 0, step, 1.0)[..., None]
    codes = np.where(step[..., None] > 0, np.clip(np.round(t), 0, 15), 0.0)
    out = (
        codes.reshape(-1, BLOCKS * BLOCK).astype(np.uint8),
        scales.astype(np.uint8),
        mins.astype(np.uint8),
        d[:, 0],
        dmin[:, 0],
    )
    if not report:
        return out

    def near(x, margin):
        return np.abs(x - np.floor(x) - 0.5) < margin

    def f16_unstable(v):
        return (v * (1 + 1e-5)).astype(np.float16) != (v * (1 - 1e-5)).astype(np.float16)

    fields_bad = bool(
        near(ls_raw, 0.01).any()
        or near(lm_raw, 0.01).any()
        or f16_unstable(max_scale / 63.0).any()
        or f16_unstable(max_min / 63.0).any()
    )
    codes_bad = near(t, margin) & (t > 0) & (t < 15) & (step[..., None] > 0)
    return out, fields_bad, codes_bad, t, step


def tie_free(w, margin=0.02):
    """A tie-free copy of the superblocks w, or None when they must be drawn again."""
    w = np.array(w, dtype=np.float32)
    _, fields_bad, codes_bad, t, step = model_encode(w, report=True, margin=margin)
    if fields_bad:
        return None
    blocks = w.reshape(-1, BLOCKS, BLOCK)
    lo, hi = blocks.min(axis=-1), blocks.max(axis=-1)
    moved = blocks + np.where(codes_bad, (np.round(t) - t) * 0.25 * step[..., None], 0.0)
    moved = moved.astype(np.float32)
    if not (np.array_equal(moved.min(axis=-1), lo) and np.array_equal(moved.max(axis=-1), hi)):
        return None
    moved = moved.reshape(w.shape)
    _, fields_bad, codes_bad, _, _ = model_encode(moved, report=True, margin=margin)
    return None if fields_bad or codes_bad.any() else moved


def superblocks(ctx, make, count, margin=0.02):
    """`count` tie-free superblocks; `make()` draws one candidate of shape (256,)."""
    rows = []
    for _ in range(count):
        for _ in range(400):
            row = tie_free(np.asarray(make(), dtype=np.float32).reshape(1, -1), margin)
            if row is not None:
                rows.append(row[0])
                break
        else:
            raise RuntimeError("could not draw a tie-free superblock")
    return np.stack(rows)


def varied(ctx, *, magnitude=1.0, offset=1.0):
    """A superblock whose blocks differ in range and in center."""

    def make():
        width = ctx.uniform(BLOCKS, 1, low=0.1, high=1.0) * magnitude
        center = ctx.uniform(BLOCKS, 1, low=-offset, high=offset) * magnitude
        return (ctx.randn(BLOCKS, BLOCK) * width + center).reshape(-1)

    return make


def decode_with(ctx, module, fw, fields):
    """Call q4k_decode with uint8 codes and fields and float16 d and dmin."""
    codes, scales, mins, d, dmin = fields
    args = [fw.tensor(np.asarray(a, dtype=np.uint8)) for a in (codes, scales, mins)]
    args += [fw.tensor(np.asarray(a), dtype=np.float16) for a in (d, dmin)]
    return fw.to_numpy(module.q4k_decode(*args))


def model_decode(fields):
    codes, scales, mins, d, dmin = (np.asarray(a).astype(np.float64) for a in fields)
    step = d[:, None] * scales
    offset = dmin[:, None] * mins
    q = codes.reshape(-1, BLOCKS, BLOCK)
    return (step[..., None] * q - offset[..., None]).reshape(codes.shape)


def check_decode(ctx, fields):
    got = decode_with(ctx, ctx.solution, ctx.fw, fields)
    want = model_decode(fields)
    oracle = decode_with(ctx, ctx.oracle, ctx.oracle_fw, fields)
    atol = 1e-6 * float(np.abs(want).max()) + 1e-12
    check.allclose(oracle, want, rtol=1e-5, atol=atol, label="oracle")
    check.shape(got, want.shape, label="q4k_decode")
    check.that(
        got.dtype == np.float32,
        f"q4k_decode: got dtype {got.dtype}, which is not the expected dtype",
        "expected float32",
    )
    check.allclose(got, want, rtol=1e-5, atol=atol, label="q4k_decode")
    return got


NAMES = ("codes", "scales", "mins", "d", "dmin")


def check_encode(ctx, w):
    """q4k_encode against the float64 model, field by field, with dtypes."""
    w = np.asarray(w, dtype=np.float32)
    want = model_encode(w)
    oracle = ctx.expected("q4k_encode", w)
    if not all(np.array_equal(a, b) for a, b in zip(oracle, want, strict=True)):
        raise RuntimeError("the oracle disagrees with the float64 model")  # an authoring error
    got = ctx.call("q4k_encode", w)
    check.that(
        isinstance(got, (tuple, list)) and len(got) == 5,
        "q4k_encode: expected a tuple (codes, scales, mins, d, dmin)",
    )
    # The header first: a wrong unit explains wrong fields, and wrong fields wrong codes.
    for index in (3, 4, 1, 2, 0):
        name, value, expected = NAMES[index], got[index], want[index]
        check.shape(value, expected.shape, label=name)
        check.that(
            value.dtype == expected.dtype,
            f"{name}: got dtype {value.dtype}, which is not the expected dtype",
            f"expected {expected.dtype}",
        )
        if index >= 3:
            check.that(
                bool(np.array_equal(value, expected)),
                f"{name}: not the expected float16 values",
                f"expected {expected.tolist()}\ngot      {value.tolist()}",
            )
        else:
            check.equal(value, expected, label=name)
    return got


def check_round_trip(ctx, w):
    fields = check_encode(ctx, w)
    return fields, check_decode(ctx, fields)


def worked_input():
    w = np.zeros((1, 256), dtype=np.float32)
    w[0, 0:4] = [-1.0, 0.25, 2.0, 0.66]
    w[0, 32:36] = [0.52, 1.0, 0.32, 0.75]
    w[0, 64:96] = -0.4
    return w


@case("the worked example: one superblock, three blocks that are not zero", public=True)
def worked_example(ctx):
    w = worked_input()
    codes, scales, mins, d, dmin = ctx.call("q4k_encode", w)
    check.allclose(d, np.array([0.003173828125]), rtol=1e-6, label="d")
    check.allclose(dmin, np.array([0.015869140625]), rtol=1e-6, label="dmin")
    check.equal(scales, np.array([[63, 21, 0, 0, 0, 0, 0, 0]]), label="scales")
    check.equal(mins, np.array([[63, 0, 25, 0, 0, 0, 0, 0]]), label="mins")
    check.equal(codes[0, 0:4], np.array([0, 6, 15, 8]), label="codes[0, 0:4]")
    check.equal(codes[0, 32:36], np.array([8, 15, 5, 11]), label="codes[0, 32:36]")
    check.equal(codes[0, 64:68], np.array([0, 0, 0, 0]), label="codes[0, 64:68]")
    _, decoded = check_round_trip(ctx, w)
    check.allclose(
        decoded[0, [0, 1, 2, 3, 32, 33, 34, 35, 64]],
        np.array([-1.0, 0.2, 2.0, 0.6, 0.5333, 1.0, 0.3333, 0.7333, -0.3968], np.float32),
        rtol=2e-3,
        label="decoded values",
    )


# Rejects ideal_block_grid, fields_rounded_down, scale_from_first_block, unit_over_64,
# sixteen_steps, skips_float16.
@case("two superblocks whose blocks differ in range and center", public=True)
def varied_blocks(ctx):
    check_round_trip(ctx, superblocks(ctx, varied(ctx), 2))


# Rejects min_not_clamped.
@case("blocks of positive weights only: the block minimum is 0, not positive", public=True)
def positive_blocks(ctx):
    def make():
        w = np.abs(ctx.randn(BLOCKS, BLOCK)) * ctx.uniform(BLOCKS, 1, low=0.2, high=1.0) + 0.3
        w[::3] = ctx.randn(3, BLOCK)  # a few blocks that do go negative
        return w.reshape(-1)

    (_, _, mins, _, _), _ = check_round_trip(ctx, superblocks(ctx, make, 2))
    check.that(
        not mins.reshape(2, BLOCKS)[:, [1, 2, 4, 5, 7]].any(),
        "mins: a block without negative weights must get minimum 0",
        f"mins = {mins.tolist()}",
    )
    all_positive = superblocks(ctx, lambda: np.abs(ctx.randn(256)) + 0.5, 2)
    (_, _, mins, _, dmin), _ = check_round_trip(ctx, all_positive)
    check.that(not mins.any() and not dmin.any(), "a superblock without negative weights: wrong")


# Rejects decode_adds_min and decode_in_uint8.
@case("q4k_decode: the largest fields, 63 and 15, and a negative offset", public=True)
def decode_extremes(ctx):
    codes = np.tile(np.arange(16, dtype=np.uint8), (2, 16))
    scales = np.array([[63, 62, 1, 0, 17, 33, 48, 5], [0, 0, 0, 0, 0, 0, 0, 63]], dtype=np.uint8)
    mins = np.array([[63, 0, 31, 9, 0, 60, 2, 63], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=np.uint8)
    d = np.array([0.0123, 2.5], dtype=np.float16)
    dmin = np.array([0.5, 0.004], dtype=np.float16)
    got = check_decode(ctx, (codes, scales, mins, d, dmin))
    check.that(
        abs(float(got[0, 15]) - (float(d[0]) * 63 * 15 - float(dmin[0]) * 63)) < 1e-3,
        "q4k_decode: wrong value for scale 63, code 15, minimum 63",
        f"got {float(got[0, 15])}",
    )


# Rejects zero_superblock_nan.
@case("zero and constant blocks, and a superblock that is entirely zero", public=True)
def degenerate_blocks(ctx):
    def make():
        w = ctx.randn(BLOCKS, BLOCK)
        w[1] = 0.0
        w[3] = -0.75  # constant and negative: scale 0, carried by the minimum alone
        w[6] = 0.4  # constant and positive: one step, from 0 to 0.4
        return w.reshape(-1)

    w = superblocks(ctx, make, 2)
    w[1] = 0.0
    (codes, scales, mins, d, dmin), decoded = check_round_trip(ctx, w)
    check.that(
        not (codes[1].any() or scales[1].any() or mins[1].any() or d[1] or dmin[1]),
        "an all-zero superblock must be encoded as zeros",
    )
    check.finite(decoded, label="q4k_decode")
    check.that(not decoded[1].any(), "an all-zero superblock must decode to zeros")


# Rejects skips_float16. The float16 rounding moves a step by up to 1 part in 2048, and
# (w + offset) / step by up to 0.007, so this case needs weights closer to a tie than
# the others allow: 0.003, still hundreds of times the float32 rounding error.
@case("the codes are computed with the float16 units, not with their float32 originals")
def float16_units(ctx):
    for _ in range(200):
        w = superblocks(ctx, varied(ctx), 2, margin=0.003)
        exact = model_encode(w, exact_units=True)[0]
        if int((exact != model_encode(w)[0]).sum()) >= 3:
            break
    else:  # an authoring error
        raise RuntimeError("no input separates the float16 units from the float32 ones")
    check_round_trip(ctx, w)


@case("small weights: d and dmin are subnormal float16 numbers")
def small_weights(ctx):
    check_round_trip(ctx, superblocks(ctx, varied(ctx, magnitude=0.02), 2))
    check_round_trip(ctx, superblocks(ctx, varied(ctx, magnitude=60.0), 2))


@case("one block much larger than the others: small blocks get scale 0 or 1")
def dominant_block(ctx):
    def make():
        w = ctx.randn(BLOCKS, BLOCK) * 0.01
        w[int(ctx.rng.integers(BLOCKS))] = ctx.randn(BLOCK) * 3.0
        w[int(ctx.rng.integers(BLOCKS))] = ctx.randn(BLOCK) * 0.2
        return w.reshape(-1)

    check_round_trip(ctx, superblocks(ctx, make, 4))


@case("q4k_decode: random fields over their whole range")
def decode_random(ctx):
    for _ in range(3):
        fields = (
            ctx.randint(0, 16, (4, 256)).astype(np.uint8),
            ctx.randint(0, 64, (4, 8)).astype(np.uint8),
            ctx.randint(0, 64, (4, 8)).astype(np.uint8),
            (10.0 ** ctx.uniform(4, low=-5, high=0)).astype(np.float16),
            (10.0 ** ctx.uniform(4, low=-5, high=0)).astype(np.float16),
        )
        check_decode(ctx, fields)


@case("the round-trip error stays within the two-level bound; fields fit their bits")
def error_bound(ctx):
    def prop(rng):
        magnitude = float(10.0 ** rng.uniform(-1, 1.5))
        w = superblocks(ctx, varied(ctx, magnitude=magnitude, offset=float(rng.uniform(0, 2))), 2)
        (codes, scales, mins, d, dmin), decoded = check_round_trip(ctx, w)
        check.that(
            int(codes.max()) <= 15 and int(scales.max()) <= 63 and int(mins.max()) <= 63,
            "a code above 15 or a block scale or minimum above 63 does not fit its bits",
        )
        d64, dmin64 = d.astype(np.float64)[:, None], dmin.astype(np.float64)[:, None]
        step = d64 * scales
        bound = (0.5 * step + 7.5 * d64 + 0.5 * dmin64) * 1.05 + 1e-6
        error = np.abs(decoded.astype(np.float64) - w).reshape(2, BLOCKS, BLOCK).max(axis=-1)
        worst = np.unravel_index(np.argmax(error - bound), error.shape)
        check.that(
            bool((error <= bound).all()),
            "a decoded weight is further from the original than the two-level bound allows",
            f"superblock {worst[0]}, block {worst[1]}: |error| = {error[worst]:.6g}, "
            f"bound = {bound[worst]:.6g}",
        )

    check.forall(prop, trials=6)
