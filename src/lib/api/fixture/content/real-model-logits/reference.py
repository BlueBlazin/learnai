{{import}}


def next_token_logits(hidden: {{T}}, norm_weight: {{T}}, unembed: {{T}}) -> {{T}}:
    """Return logits of shape (V,) for the last position."""
    last = hidden[-1]
    scale = torch.rsqrt((last * last).mean() + 1e-5)
    return unembed @ (last * scale * norm_weight)
