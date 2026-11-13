{{import}}


def causal_mask(lengths: {{T}}, max_len: int) -> {{T}}:
    """Return a (B, 1, max_len, max_len) bool mask; True where i may attend to j."""
    raise NotImplementedError
