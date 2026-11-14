import math

import mlx.core as mx

FLOAT_BITS = 16  # scales, biases, and unquantized tensors are stored as 16-bit floats


def quantize_dequantize(w: mx.array, bits: int, group_size: int) -> mx.array:
    """Quantize a matrix group by group the way MLX does, and return what the codes decode to.

    w: float32, shape (out, in); `in` is a multiple of group_size. Groups are runs of
        group_size consecutive inputs within a row.
    bits: 2 to 8; codes are 0 .. 2**bits - 1.
    Returns float32, shape (out, in): code * scale + bias for every weight.
    """
    raise NotImplementedError


def quantize_report(weights: dict[str, mx.array], bits: int, group_size: int) -> dict:
    """Quantize every matrix of a model and report what it costs.

    weights: mapping from tensor names to float32 tensors. The 2-D ones are quantized;
        the others (1-D norm gains) are kept as 16-bit floats.
    Returns a dict of Python values:
        "sqnr_db": {name: SQNR in dB} for every 2-D tensor,
        "model_sqnr_db": 10 log10(total signal power / total noise power) over them,
        "bits_per_weight": stored bits of all tensors / number of parameters of all tensors,
        "worst": the names of the three matrices with the lowest SQNR, lowest first.
    """
    raise NotImplementedError
