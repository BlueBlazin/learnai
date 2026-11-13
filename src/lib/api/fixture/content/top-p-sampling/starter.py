{{import}}


def sample_top_p(logits: {{T}}, p: float, u: {{T}}) -> {{T}}:
    """
    logits: (B, V) float32.    p in (0, 1].    u: (B,) uniforms in [0, 1).
    Returns (B,) int64 token ids.
    """
    raise NotImplementedError
