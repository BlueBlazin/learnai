import mlx.core as mx


def mxfp4_encode(x: mx.array) -> tuple[mx.array, mx.array]:
    """Encode x in MXFP4: blocks of 32 values share one power-of-two scale.

    x: float32 array of shape (out, in), in a multiple of 32.
    Returns (codes, exps): uint8 codes of shape (out, in), each a 4-bit E2M1 float
    (bit 3 sign, bits 2-1 exponent, bit 0 mantissa), and uint8 scale bytes of shape
    (out, in / 32), each the block's shared exponent plus 127.
    """
    raise NotImplementedError


def mxfp4_decode(codes: mx.array, exps: mx.array) -> mx.array:
    """Decode MXFP4 blocks: 2**(exps - 127) times the value of each 4-bit code.

    codes: uint8 array of shape (out, in), values 0 to 15. exps: uint8 array of
    shape (out, in / 32).
    Returns a float32 array of shape (out, in).
    """
    raise NotImplementedError
