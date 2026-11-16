"""Tests for bf16-decode.

Everything is compared bit for bit. Float results are reinterpreted as uint32 before
the comparison, so that NaN payloads, the sign of zero, and subnormals all count, and
failure details print bit patterns in hexadecimal.
"""

import numpy as np

from learnai.judge import case, check


def f32(patterns):
    """float32 array with the given 32-bit patterns."""
    return np.array(patterns, dtype=np.uint32).view(np.float32)


def u16(patterns):
    return np.array(patterns, dtype=np.uint16)


def is_nan32(patterns):
    return (np.asarray(patterns, dtype=np.uint32) & 0x7FFFFFFF) > 0x7F800000


def same_patterns(got, want, inputs, *, label, width):
    """Fail unless the integer arrays `got` and `want` are equal; show hex in the detail."""
    bad = np.flatnonzero(got.ravel() != want.ravel())
    if bad.size == 0:
        return
    in_width = 12 - width  # 16-bit inputs print as 4 hex digits, 32-bit ones as 8
    rows = [
        f"  input 0x{int(inputs.ravel()[i]):0{in_width}X}: "
        f"expected 0x{int(want.ravel()[i]):0{width}X}, got 0x{int(got.ravel()[i]):0{width}X}"
        for i in bad[:8]
    ]
    check.that(
        False,
        f"{label}: {bad.size} of {got.size} bit patterns are wrong",
        "\n".join(["first mismatches (input bits, expected bits, got bits):", *rows]),
    )


def check_decode(ctx, bits, want_bits=None):
    """bf16_to_f32(bits) must be float32 with exactly the expected 32-bit patterns."""
    got = np.ascontiguousarray(ctx.call("bf16_to_f32", bits))
    check.shape(got, bits.shape, label="bf16_to_f32")
    check.that(
        got.dtype == np.float32,
        "bf16_to_f32 must return a float32 tensor",
        f"got dtype {got.dtype}",
    )
    if want_bits is None:
        want_bits = np.ascontiguousarray(ctx.expected("bf16_to_f32", bits)).view(np.uint32)
    want_bits = np.asarray(want_bits, dtype=np.uint32).reshape(bits.shape)
    same_patterns(got.view(np.uint32), want_bits, bits, label="bf16_to_f32", width=8)
    return got


def check_encode(ctx, x, want=None):
    """f32_to_bf16(x) must be uint16 with exactly the expected patterns."""
    got = np.asarray(ctx.call("f32_to_bf16", x))
    check.shape(got, x.shape, label="f32_to_bf16")
    check.that(
        got.dtype == np.uint16,
        "f32_to_bf16 must return a uint16 tensor",
        f"got dtype {got.dtype}",
    )
    if want is None:
        want = np.asarray(ctx.expected("f32_to_bf16", x))
    want = np.asarray(want, dtype=np.uint16).reshape(x.shape)
    same_patterns(got, want, np.ascontiguousarray(x).view(np.uint32), label="f32_to_bf16", width=4)
    return got


@case("the worked example: decode five patterns, encode six values", public=True)
def worked_example(ctx):
    bits = u16([0x3F80, 0xC020, 0x4049, 0x7F80, 0x0001])
    check_decode(ctx, bits, [0x3F800000, 0xC0200000, 0x40490000, 0x7F800000, 0x00010000])
    x = np.array([1.0, -2.5, 3.14159265, 0.1, 1.00390625, 1.01171875], dtype=np.float32)
    check_encode(ctx, x, [0x3F80, 0xC020, 0x4049, 0x3DCD, 0x3F80, 0x3F82])


# Rejects ieee_half: 0x3C00 is 1.0 in half precision and 0.0078125 in bfloat16.
@case("decoding: ordinary values, signed zeros, infinities, and subnormals", public=True)
def decode_values(ctx):
    bits = u16(
        [
            [0x3C00, 0xBC00, 0x3F00, 0x4000],  # 2**-7, -2**-7, 0.5, 2.0
            [0x0000, 0x8000, 0x7F80, 0xFF80],  # +0.0, -0.0, +inf, -inf
            [0x0001, 0x007F, 0x0080, 0x7F7F],  # subnormals, smallest normal, largest
            [0x3DCD, 0x42F7, 0xC2C8, 0x8001],
        ]
    )
    check_decode(ctx, bits)


# Rejects truncates.
@case("encoding rounds to the nearest neighbor, up or down", public=True)
def rounds_to_nearest(ctx):
    x = np.array([0.1, 0.2, 0.3, -0.7, 1e-3, 123.456, -9876.5, 1e10, 3e-20], dtype=np.float32)
    check_encode(ctx, x)
    # One step below a tie, and one step above it, on both sides of zero.
    check_encode(ctx, f32([0x3F807FFF, 0x3F808001, 0xBF807FFF, 0xBF808001, 0x3F81FFFF]))


# Rejects rounds_half_up: half of these ties must go down.
@case("exact ties go to the even neighbor", public=True)
def ties_to_even(ctx):
    ties = f32(
        [
            [0x3F808000, 0x3F818000, 0x3F828000, 0x3F838000],
            [0xBF808000, 0xBF818000, 0x41208000, 0x41218000],
            [0x00008000, 0x00018000, 0x80008000, 0x80018000],
        ]
    )
    check_encode(ctx, ties)


# Rejects carry_dropped: every mantissa here rounds up to 2.0.
@case("rounding up carries into the exponent, and into infinity")
def carry_into_exponent(ctx):
    x = f32(
        [
            0x3FFFFFFF,  # just below 2.0 -> 2.0
            0x3FFF8000,  # a tie whose even neighbor is 2.0
            0xC07FC000,  # just short of -4.0 -> -4.0
            0x40FF8001,
            0x007FFFFF,  # the largest subnormal -> the smallest normal
            0x7F7FFFFF,  # the largest float32 -> +inf
            0xFF7F8000,  # a tie whose even neighbor is -inf
            0x7F7F7FFF,  # stays finite
        ]
    )
    check_encode(ctx, x)


# Rejects nan_unhandled and nan_sign_dropped.
@case("NaN encodes as 0x7FC0 or 0xFFC0, whatever its payload")
def nan_encoding(ctx):
    nans = [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF]
    nans += [0x7F80FFFF, 0xFFA00000, 0x7FFF8000, 0xFFFF7FFF, 0x7F808000, 0xFFBFFFFF]
    got = check_encode(ctx, f32(nans), [0xFFC0 if p >> 31 else 0x7FC0 for p in nans])
    check.shape(got, (len(nans),), label="f32_to_bf16")
    # Infinities are not NaN and stay infinities.
    check_encode(ctx, f32([0x7F800000, 0xFF800000]), [0x7F80, 0xFF80])


@case("NaN patterns decode with their sign and payload intact")
def nan_decoding(ctx):
    bits = u16([0x7FC0, 0xFFC0, 0x7F81, 0xFF81, 0x7FFF, 0xFFFF, 0x7FA5, 0xFFDA])
    check_decode(ctx, bits, bits.astype(np.uint32) << 16)


@case("subnormals and signed zeros encode exactly")
def tiny_values(ctx):
    x = f32(
        [
            0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00007FFF, 0x00008001,
            0x00010000, 0x0001C000, 0x00024000, 0x007F0000, 0x807E8000, 0x00800000,
        ]
    )  # fmt: skip
    check_encode(ctx, x)


@case("decoding all 65,536 patterns")
def decode_everything(ctx):
    bits = np.arange(65536, dtype=np.uint32).astype(np.uint16).reshape(256, 256)
    check_decode(ctx, bits, bits.astype(np.uint32) << 16)


@case("round trip: every non-NaN pattern survives decode, then encode, unchanged")
def round_trip_exhaustive(ctx):
    bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    values = np.ascontiguousarray(ctx.call("bf16_to_f32", bits))
    check.shape(values, bits.shape, label="bf16_to_f32")
    back = np.asarray(ctx.call("f32_to_bf16", values))
    check.shape(back, bits.shape, label="f32_to_bf16(bf16_to_f32(bits))")
    nan = (bits & 0x7FFF) > 0x7F80
    want = np.where(nan, (bits & 0x8000) | 0x7FC0, bits).astype(np.uint16)
    same_patterns(
        back.astype(np.uint16), want, bits, label="f32_to_bf16(bf16_to_f32(bits))", width=4
    )


@case("random 32-bit patterns of every kind, in 1-D, 2-D, and 3-D shapes")
def random_patterns(ctx):
    for shape in ((2048,), (32, 64), (4, 8, 16)):
        patterns = ctx.rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
        check_encode(ctx, patterns.view(np.float32))
        check_decode(ctx, (patterns >> 16).astype(np.uint16))


@case("the encoded value is never farther from x than half a bfloat16 step")
def nearest_property(ctx):
    def prop(rng):
        # Finite values below the point where rounding overflows to infinity.
        # (Clearing the top exponent bit of the others keeps the shape fixed.)
        patterns = rng.integers(0, 2**32, size=2048, dtype=np.uint64).astype(np.uint32)
        patterns = np.where((patterns & 0x7FFFFFFF) < 0x7F7F8000, patterns, patterns & 0xBFFFFFFF)
        x = patterns.view(np.float32)
        encoded = np.asarray(ctx.call("f32_to_bf16", x))
        check.shape(encoded, x.shape, label="f32_to_bf16")
        # Decode here, with numpy, so that this case judges f32_to_bf16 alone.
        decoded = (encoded.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        check.finite(decoded, label="the value encoded by f32_to_bf16")
        # A bfloat16 step is 2**16 float32 steps: 2**(e - 134), with e the biased exponent
        # (1 for subnormals).
        exponent = np.maximum((patterns >> 23) & 0xFF, 1).astype(np.float64)
        half_step = 2.0 ** (exponent - 135)
        error = np.abs(decoded - x.astype(np.float64))
        worst = int(np.argmax(error / half_step))
        check.that(
            bool(np.all(error <= half_step)),
            "f32_to_bf16 returned a value that is not the nearest bfloat16",
            f"x has bits 0x{int(patterns[worst]):08X}; got 0x{int(encoded[worst]):04X}, "
            f"which is {error[worst] / half_step[worst]:.3f} half-steps away",
        )

    check.forall(prop, trials=3)
