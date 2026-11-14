# BUG: adds up the stored bits in a 32-bit integer tensor: right for a small model, but Granite's 3.0e9 bits (374 MB) wrap around past 2**31.
import math

import torch

FLOAT_BITS = 16  # scales, biases, and unquantized tensors are stored as 16-bit floats


def quantize_dequantize(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code
    # Every row is cut into consecutive runs of group_size inputs.
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1, keepdim=True)
    hi = groups.amax(dim=-1, keepdim=True)
    # The grid starts at the end of the range that is further from zero (code 0) and
    # steps toward the other end: upward from the minimum, or downward from the maximum.
    from_low = lo.abs() > hi.abs()
    step = torch.clamp((hi - lo) / levels, min=1e-7)  # the floor keeps a constant group finite
    scale = torch.where(from_low, step, -step)
    edge = torch.where(from_low, lo, hi)
    # Zero lies -edge / scale steps from the edge. Rounding that to an integer and
    # recomputing the scale puts a grid point exactly on zero.
    q0 = torch.round(edge / scale)
    scale = torch.where(q0 != 0, edge / torch.where(q0 != 0, q0, 1.0), scale)
    bias = torch.where(q0 != 0, edge, 0.0)
    codes = torch.clamp(torch.round((groups - bias) / scale), 0, levels)
    return (codes * scale + bias).reshape(out_dim, in_dim)


def quantize_report(weights: dict[str, torch.Tensor], bits: int, group_size: int) -> dict:
    sqnr_db = {}
    signal_total = noise_total = 0.0
    stored_bits = torch.zeros((), dtype=torch.int32)
    parameters = 0
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
        "bits_per_weight": stored_bits.item() / parameters,
        "worst": sorted(sqnr_db, key=sqnr_db.get)[:3],  # lowest SQNR first
    }
