{{import}}


def apply_rope(x: {{T}}, positions: {{T}}, base: float = 10000.0) -> {{T}}:
    half = x.shape[-1] // 2
    freqs = base ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * freqs[None, :]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return out.flatten(-2)
