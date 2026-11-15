import mlx.core as mx

BLOCK = 32  # elements that share one scale
EMAX = 2  # the largest exponent of an E2M1 element: its largest power of two is 2**2
BIAS = 127  # exponent bias of the E8M0 scale, the same as float32's

# The eight magnitudes of an E2M1 element, indexed by its three low bits (exponent, mantissa).
MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _exponent_field(m: mx.array) -> mx.array:
    # The biased exponent of a positive normal float32, read from its bits:
    # floor(log2(m)) + 127, with no logarithm and no rounding.
    return m.view(mx.int32) >> 23


def _power_of_two(field: mx.array) -> mx.array:
    # The float32 whose exponent field is `field` and whose mantissa is 0: exactly
    # 2**(field - 127). An E8M0 scale is this field, stored on its own.
    # (mx.power(2.0, e) is not exact for every integer e.)
    return (field << 23).view(mx.float32)


def mxfp4_encode(x: mx.array) -> tuple[mx.array, mx.array]:
    out_dim, in_dim = x.shape
    blocks = x.reshape(out_dim, in_dim // BLOCK, BLOCK)
    largest = mx.max(mx.abs(blocks), axis=-1)
    # The largest power of two not above the block maximum, divided by 2**EMAX.
    # An all-zero block has no such power; it gets the scale 1.
    field = mx.where(largest > 0, _exponent_field(largest) - EMAX, BIAS)
    scale = _power_of_two(field)
    # Dividing by a power of two is exact, so a tie in the input is a tie here.
    # Magnitudes above the largest element value, 6, saturate.
    a = mx.minimum(mx.abs(blocks) / scale[..., None], 6.0)
    # The grid step is 0.5 below 2, 1 on [2, 4), and 2 on [4, 6]: round in units of the
    # step. round() rounds half to even, and an even count is a code with mantissa bit 0.
    magnitude = mx.where(
        a < 2.0,
        mx.round(a * 2.0),
        mx.where(a < 4.0, mx.round(a) + 2.0, mx.round(a / 2.0) + 4.0),
    )
    # The sign bit is the sign of the input, also when the magnitude rounds to 0.
    codes = magnitude + 8.0 * (blocks < 0)
    return codes.reshape(out_dim, in_dim).astype(mx.uint8), field.astype(mx.uint8)


def mxfp4_decode(codes: mx.array, exps: mx.array) -> mx.array:
    codes = codes.astype(mx.int32)
    # Three low bits: exponent and mantissa, through the table. Bit 3: the sign.
    magnitude = mx.take(mx.array(MAGNITUDES), codes & 7)
    sign = 1.0 - 2.0 * (codes >> 3)
    # Widen the scale byte before any arithmetic on it; then one scale per block of 32.
    scale = _power_of_two(exps.astype(mx.int32))
    return sign * magnitude * mx.repeat(scale, BLOCK, axis=1)
