import mlx.core as mx


def one_hot_rows(ids: mx.array, V: int) -> mx.array:
    # put_along_axis along axis 1 writes out[b, index[b, j]] = 1. The index needs as
    # many axes as the array written into: (B,) -> (B, 1). The value must be an array.
    out = mx.zeros((ids.shape[0], V), dtype=mx.float32)
    return mx.put_along_axis(out, ids[:, None], mx.array(1.0), axis=1)


def token_counts(ids: mx.array, V: int) -> mx.array:
    # rows is (B, 1) and broadcasts against ids (B, T): one (row, column) pair per
    # token. .at[...].add accumulates the values that land on one entry;
    # put_along_axis would keep a single 1 for a repeated token.
    rows = mx.arange(ids.shape[0])[:, None]
    return mx.zeros((ids.shape[0], V), dtype=mx.int32).at[rows, ids].add(1)


def write_at(x: mx.array, idx: mx.array, values: mx.array) -> mx.array:
    # put_along_axis returns a new array: out[b, idx[b, k]] = values[b, k], and every
    # other entry is copied from x.
    return mx.put_along_axis(x, idx, values, axis=1)
