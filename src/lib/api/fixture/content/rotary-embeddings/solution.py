{{import}}


def apply_rope(x: {{T}}, positions: {{T}}, base: float = 10000.0) -> {{T}}:
    half = x.shape[-1] // 2
    out = torch.empty_like(x)
    # First attempt: correct, but one Python iteration per position.
    for t, m in enumerate(positions.tolist()):
        for i in range(half):
            theta = m * base ** (-i / half)
            c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
            a, b = x[..., t, 2 * i], x[..., t, 2 * i + 1]
            out[..., t, 2 * i] = a * c - b * s
            out[..., t, 2 * i + 1] = a * s + b * c
    return out
