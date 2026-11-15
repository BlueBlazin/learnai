import mlx.core as mx

from mlx import nn


def split_fused(w: mx.array, parts: list[int]) -> list[mx.array]:
    # Weights are (out, in): the projections are stacked along the output axis, so
    # each one is a block of consecutive rows. `parts` holds sizes (mx.split would
    # read a list as the offsets at which to cut), so walk the offsets by hand.
    out, start = [], 0
    for size in parts:
        out.append(w[start : start + size])
        start += size
    return out


def fuse(ws: list[mx.array]) -> mx.array:
    # Stacking rows concatenates the outputs: x @ fuse(ws)^T = cat(x @ w^T for w in ws).
    return mx.concatenate(ws, axis=0)


def swiglu_fused(x: mx.array, w_in: mx.array, w_down: mx.array) -> mx.array:
    # One matrix product computes both branches; the first half of the features is
    # the gate (it gets the SiLU), the second half is the up branch.
    gate, up = mx.split(x @ w_in.T, 2, axis=-1)
    return (nn.silu(gate) * up) @ w_down.T
