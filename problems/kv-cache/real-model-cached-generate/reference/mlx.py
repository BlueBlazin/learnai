import mlx.core as mx

from learnai.kit import RealConfig, ops
from mlx import nn

# The cache: for each layer, (keys, values) of every token read so far, each of shape
# (n_kv_heads, length, head_dim). Keys are stored already rotated.
Cache = list[tuple[mx.array, mx.array]]


def _forward(
    weights: dict[str, mx.array], cfg: RealConfig, tokens: mx.array, cache: Cache | None
) -> tuple[mx.array, Cache]:
    """Run `tokens` (1-D) after the cached ones: logits after the last, and the new cache."""
    seq = tokens.shape[0]
    past = 0 if cache is None else cache[0][0].shape[1]
    half = cfg.head_dim // 2
    group = cfg.n_heads // cfg.n_kv_heads
    # The new tokens continue where the cache ends: positions past .. past + seq - 1.
    positions = past + mx.arange(seq)
    inv_freq = cfg.rope_theta ** (-mx.arange(half).astype(mx.float32) / half)
    angles = positions.astype(mx.float32)[:, None] * inv_freq  # (T, half)
    cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    # The token at position p sees entries 0..p: all cached ones, and the new ones up
    # to itself. For a single new token every entry is visible.
    visible = mx.arange(past + seq)[None, :] <= positions[:, None]  # (T, past + T)

    def rope(z: mx.array) -> mx.array:
        rotated = mx.concatenate([-z[..., half:], z[..., :half]], axis=-1)  # (a, b) -> (-b, a)
        return z * cos + rotated * sin

    def heads(z: mx.array, n: int) -> mx.array:
        return z.reshape(seq, n, cfg.head_dim).transpose(1, 0, 2)  # (n, T, head_dim)

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier
    new_cache = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        # A key is rotated once, for its own position, and cached rotated.
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        if cache is not None:
            # concatenate builds new arrays: the cache that was passed in stays as it was.
            k = mx.concatenate([cache[i][0], k], axis=1)
            v = mx.concatenate([cache[i][1], v], axis=1)
        new_cache.append((k, v))  # at KV-head width
        # Each KV head serves `group` consecutive query heads: expanded for use only.
        scores = q @ mx.repeat(k, group, axis=0).transpose(0, 2, 1)
        scores = mx.where(visible, scores * cfg.attention_scale, -mx.inf)
        mixed = ops.softmax(scores) @ mx.repeat(v, group, axis=0)
        mixed = mixed.transpose(1, 0, 2).reshape(seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate, then up
        hidden = nn.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    # Only the last position chooses a token: one row through the norm and the head.
    last = ops.rms_norm(x[-1], weights["model.norm.weight"], cfg.norm_eps)
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (head @ last) / cfg.logits_scaling, new_cache


def prefill(
    weights: dict[str, mx.array], cfg: RealConfig, prompt_ids: mx.array
) -> tuple[mx.array, Cache]:
    return _forward(weights, cfg, prompt_ids, None)


def decode_step(
    weights: dict[str, mx.array], cfg: RealConfig, token: int, cache: Cache
) -> tuple[mx.array, Cache]:
    return _forward(weights, cfg, mx.array([token]), cache)


def generate(
    weights: dict[str, mx.array], cfg: RealConfig, prompt_ids: mx.array, max_new_tokens: int
) -> list[int]:
    logits, cache = prefill(weights, cfg, prompt_ids)  # the whole prompt, once
    tokens = []
    for step in range(max_new_tokens):
        token = int(mx.argmax(logits))  # reading the token evaluates this step's graph
        tokens.append(token)
        if step + 1 < max_new_tokens:  # the last token's logits would never be read
            logits, cache = decode_step(weights, cfg, token, cache)  # one token per step
    return tokens
