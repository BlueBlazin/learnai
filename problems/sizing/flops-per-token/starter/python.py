def prefill_flops(cfg: dict, t: int) -> int:
    """FLOPs to process a prompt of `t` tokens in one pass and produce one row of logits.

    cfg: a dict with (at least) the keys vocab_size, d_model, n_layers, n_heads,
        n_kv_heads, head_dim, d_ff (ints) and mlp_type ("swiglu": three matrices,
        or "gelu": two).
    t: number of prompt tokens, >= 1.
    Convention: multiplying an m x n matrix by an n x p matrix costs 2 * m * n * p.
    Counted: the q, k, v, o projections, the attention scores and the mixing of
    values (a dense t x t attention), the MLP matrices, and the head projection for
    the last position only. Everything else is free.
    Returns an int.
    """
    raise NotImplementedError


def decode_flops(cfg: dict, context: int) -> int:
    """FLOPs to generate one token with a KV cache.

    context: number of positions the new token attends to, itself included, >= 1.
    Same convention and terms as prefill_flops, for a single position whose one
    query per head attends over `context` cached keys and values.
    Returns an int.
    """
    raise NotImplementedError
