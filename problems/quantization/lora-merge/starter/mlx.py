import mlx.core as mx


def merge(w: mx.array, a: mx.array, b: mx.array, alpha: float) -> mx.array:
    """Fold the adapter into the weight: w + (alpha / r) * b @ a.

    w: float32 (out, in). a: float32 (r, in). b: float32 (out, r). alpha: Python float.
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError


def unmerge(w_merged: mx.array, a: mx.array, b: mx.array, alpha: float) -> mx.array:
    """Undo merge: recover w from merge(w, a, b, alpha), given the same adapter.

    w_merged: float32 (out, in). a: float32 (r, in). b: float32 (out, r).
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError


def lora_forward(x: mx.array, w: mx.array, a: mx.array, b: mx.array, alpha: float) -> mx.array:
    """The adapted layer without merging: x @ w.T + (alpha / r) * (x @ a.T) @ b.T.

    x: float32 (..., in), one to three dimensions. w, a, b, alpha: as in merge.
    Returns a float32 tensor of shape (..., out).
    """
    raise NotImplementedError
