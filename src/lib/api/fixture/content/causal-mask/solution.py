{{import}}


def causal_mask(lengths: {{T}}, max_len: int) -> {{T}}:
    """Return a (B, 1, max_len, max_len) bool mask; True where i may attend to j."""
    tri = torch.ones(max_len, max_len, dtype=torch.bool).tril()
    keys = torch.arange(max_len) < lengths[:, None]
    return (tri[None] & keys[:, None, :])[:, None]
