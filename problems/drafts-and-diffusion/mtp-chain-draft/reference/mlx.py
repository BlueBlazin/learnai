import mlx.core as mx

from learnai.kit import DecoderConfig, ops


def mtp_chain_draft(
    params: dict[str, mx.array],
    cfg: DecoderConfig,
    tokens: mx.array,
    next_token: int,
    hidden: mx.array,
    n_draft: int,
) -> tuple[mx.array, mx.array]:
    seq = tokens.shape[0]
    embedding = params["model.embed_tokens.weight"]
    # The module has no head of its own: it borrows the main model's.
    head = embedding if cfg.tie_embeddings else params["lm_head.weight"]
    causal = mx.tril(mx.ones((seq, seq))) > 0

    def heads(t: mx.array, n: int) -> mx.array:
        # (T, n * head_dim) -> (1, n, T, head_dim)
        return t.reshape(seq, n, cfg.head_dim).swapaxes(0, 1)[None]

    # The module's inputs run one token ahead of the main model's: row t pairs the hidden
    # state at position t with the token at position t + 1.
    ids = mx.concatenate((tokens[1:], mx.array([next_token], dtype=tokens.dtype)))
    h = hidden  # stage 0 reads the main model; every later stage reads the stage before
    drafted, logits = [], []
    for j in range(n_draft):
        p = f"mtp.layers.{j}."
        e = ops.rms_norm(embedding[ids], params[p + "enorm.weight"], cfg.norm_eps)
        g = ops.rms_norm(h, params[p + "hnorm.weight"], cfg.norm_eps)
        x = mx.concatenate((e, g), axis=-1) @ params[p + "eh_proj.weight"].T  # embedding first

        # One decoder block, causal, at the positions of the tokens it was given.
        b = p + "mtp_block."
        positions = mx.arange(j + 1, j + 1 + seq)
        u = ops.rms_norm(x, params[b + "input_layernorm.weight"], cfg.norm_eps)
        q = heads(u @ params[b + "self_attn.q_proj.weight"].T, cfg.n_heads)
        q = ops.rope(q, positions, cfg.rope_base)
        k = heads(u @ params[b + "self_attn.k_proj.weight"].T, cfg.n_kv_heads)
        k = ops.rope(k, positions, cfg.rope_base)
        v = heads(u @ params[b + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        attended = ops.attention(q, k, v, causal)[0].swapaxes(0, 1).reshape(seq, cfg.d_model)
        x = x + attended @ params[b + "self_attn.o_proj.weight"].T
        u = ops.rms_norm(x, params[b + "post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(
            u,
            params[b + "mlp.gate_proj.weight"],
            params[b + "mlp.up_proj.weight"],
            params[b + "mlp.down_proj.weight"],
        )

        h = ops.rms_norm(x, params[p + "post_norm.weight"], cfg.norm_eps)
        scores = h[-1] @ head.T  # only the last row is decoded
        token = mx.argmax(scores)
        drafted.append(token)
        logits.append(scores)
        ids = mx.concatenate(
            (ids[1:], token.reshape(1).astype(ids.dtype))
        )  # roll: the next stage sees this draft
    return mx.stack(drafted), mx.stack(logits)
