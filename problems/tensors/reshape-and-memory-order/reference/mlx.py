import mlx.core as mx


def group_columns(w: mx.array, g: int) -> mx.array:
    rows, cols = w.shape
    # Columns k * g .. k * g + g - 1 of a row are already next to each other in
    # row-major order, so a reshape alone draws the group boundaries.
    return w.reshape(rows, cols // g, g)


def interleave(a: mx.array, b: mx.array) -> mx.array:
    # A new last axis of size 2 puts a[..., i] and b[..., i] next to each other in
    # row-major order: (..., n, 2). Merging the last two axes then reads a0 b0 a1 b1 ...
    pairs = mx.stack((a, b), axis=-1)
    return pairs.reshape(*a.shape[:-1], 2 * a.shape[-1])


def patchify(x: mx.array, p: int) -> mx.array:
    batch, height, width = x.shape
    rows, cols = height // p, width // p
    # Split both image axes: x[b, i, u, j, v] is pixel (u, v) of patch (i, j).
    x = x.reshape(batch, rows, p, cols, p)
    # A patch is the two axes (u, v), which are not adjacent yet. Move them together,
    # (b, i, j, u, v), and only then merge (i, j) into one axis and (u, v) into another.
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(batch, rows * cols, p * p)
