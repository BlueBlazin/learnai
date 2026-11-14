import mlx.core as mx


def merge(w: mx.array, a: mx.array, b: mx.array, alpha: float) -> mx.array:
    # The rank is the number of rows of a; the update b @ a has the shape of w.
    return w + (alpha / a.shape[0]) * (b @ a)


def unmerge(w_merged: mx.array, a: mx.array, b: mx.array, alpha: float) -> mx.array:
    # Subtract exactly what merge added.
    return w_merged - (alpha / a.shape[0]) * (b @ a)


def lora_forward(x: mx.array, w: mx.array, a: mx.array, b: mx.array, alpha: float) -> mx.array:
    # Base path plus the low-rank path: project down to r dimensions with a, back up with b.
    # The (out, in) update matrix is never formed.
    return x @ w.T + (alpha / a.shape[0]) * ((x @ a.T) @ b.T)
