{{import}}


def quantized_linear(x: {{T}}, codes: {{T}}, scales: {{T}}, biases: {{T}}) -> {{T}}:
    """x: (d_in,). codes: (d_out, d_in) uint8. scales, biases: (d_out, d_in // 64)."""
    d_out, d_in = codes.shape
    groups = d_in // 64
    xg = x.reshape(groups, 64)
    qx = (codes.reshape(d_out, groups, 64) * xg).sum(axis=-1)
    return (scales * qx + biases * xg.sum(axis=-1)).sum(axis=-1)
