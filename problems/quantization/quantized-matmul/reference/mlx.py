import mlx.core as mx


def qmatmul(
    x: mx.array, codes: mx.array, scale: mx.array, bias: mx.array, group_size: int
) -> mx.array:
    out_dim, in_dim = codes.shape
    n_groups = in_dim // group_size
    # Cut the inputs and the codes into the same consecutive groups along the input axis.
    xg = x.reshape(*x.shape[:-1], n_groups, group_size)  # (..., G, g)
    cg = codes.reshape(out_dim, n_groups, group_size).astype(x.dtype)  # (O, G, g)
    # One short dot product of inputs with raw codes per (output, group).
    dots = mx.einsum("...gk,ogk->...og", xg, cg)  # (..., O, G)
    # The bias of a group multiplies the sum of that group's inputs, whatever the codes are.
    sums = xg.sum(axis=-1)  # (..., G)
    return (dots * scale).sum(axis=-1) + sums @ bias.T
