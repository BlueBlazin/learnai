import mlx.core as mx

from learnai.kit import DecoderConfig


def mtp_chain_draft(
    params: dict[str, mx.array],
    cfg: DecoderConfig,
    tokens: mx.array,
    next_token: int,
    hidden: mx.array,
    n_draft: int,
) -> tuple[mx.array, mx.array]:
    """Draft n_draft tokens with the chained MTP module in params.

    params: the checkpoint's weights by name: the main model's (model.*, and
        lm_head.weight when cfg.tie_embeddings is False) and mtp.layers.{j}.* for each
        stage (see the statement).
    cfg: vocab_size, d_model, n_heads, n_kv_heads, head_dim, rope_base, norm_eps,
        tie_embeddings.
    tokens: integer tensor, shape (T,). The tokens the main model has read.
    next_token: the token the main model then produced, a Python int.
    hidden: float32, shape (T, d_model). The main model's hidden states after its
        final norm, one row per token of `tokens`.
    n_draft: number of stages to run, 1 <= n_draft <= stages in params.
    Returns (draft_tokens, draft_logits): an integer tensor of shape (n_draft,) and
    float32 of shape (n_draft, vocab_size).
    """
    raise NotImplementedError
