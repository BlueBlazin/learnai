{{import}}


def quantized_linear(x: {{T}}, codes: {{T}}, scales: {{T}}, biases: {{T}}) -> {{T}}:
    """x: (d_in,). codes: (d_out, d_in) uint8. scales, biases: (d_out, d_in // 64)."""
    raise NotImplementedError
