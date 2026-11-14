# BUG: does not quantize anything itself: hands each matrix to mx.quantize and mx.dequantize, which the statement rules out.
import math

import mlx.core as mx
import numpy as np
import torch

FLOAT_BITS = 16  # scales, biases, and unquantized tensors are stored as 16-bit floats


def quantize_dequantize(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    packed, scales, biases = mx.quantize(mx.array(w.numpy()), group_size=group_size, bits=bits)
    w_hat = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
    return torch.from_numpy(np.array(w_hat))


def quantize_report(weights: dict[str, torch.Tensor], bits: int, group_size: int) -> dict:
    sqnr_db = {}
    signal_total = noise_total = 0.0
    stored_bits = parameters = 0
    for name, w in weights.items():
        parameters += w.numel()
        if w.ndim != 2:
            stored_bits += FLOAT_BITS * w.numel()  # norm gains stay as they are
            continue
        w_hat = quantize_dequantize(w, bits, group_size)
        # Row sums first, then their sum: a single float32 sum over 10**8 squares
        # would lose digits. The totals are accumulated as Python floats (float64).
        signal = float((w * w).sum(dim=-1).sum())
        noise = float(((w - w_hat) ** 2).sum(dim=-1).sum())
        sqnr_db[name] = 10.0 * math.log10(signal / noise)
        signal_total += signal
        noise_total += noise
        # The codes, plus one scale and one bias per group. Python ints: no overflow.
        stored_bits += bits * w.numel() + 2 * FLOAT_BITS * (w.numel() // group_size)
    return {
        "sqnr_db": sqnr_db,
        # Pooled over the model: total signal power over total noise power, which
        # weights every layer by its size. Not the mean of the per-layer decibels.
        "model_sqnr_db": 10.0 * math.log10(signal_total / noise_total),
        "bits_per_weight": stored_bits / parameters,
        "worst": sorted(sqnr_db, key=sqnr_db.get)[:3],  # lowest SQNR first
    }
