import math

import mlx.core as mx


def forward_f16(params: dict[str, mx.array], x: mx.array) -> mx.array:
    """A pre-norm feed-forward block with float16 weights, computed in float16 throughout.

    params: {"gain": (D,), "w_in": (H, D), "w_out": (D, H)}, all float16.
    x: float16 array of shape (N, D); a row may be all zeros.
    Returns x + gelu(rms_norm(x) * gain @ w_in.T) @ w_out.T as a float16 array of
    shape (N, D), with no float32 intermediate and no NaN for a zero row.
    """
    raise NotImplementedError
