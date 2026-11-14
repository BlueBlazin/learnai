import jax
import jax.numpy as jnp

from whetstone.kit import RealConfig, ops

# The cache: (keys, values, length). keys[i] and values[i] belong to layer i and have
# shape (n_kv_heads, capacity, head_dim); the first `length` slots hold the tokens read
# so far (keys already rotated), the rest are zeros that the mask hides. The capacity
# grows a bucket at a time: JAX compiles every operation once per input shape, so a
# cache that grew by one slot per token would recompile the attention at every step.
Cache = tuple[list[jax.Array], list[jax.Array], int]
BUCKET = 64


def _forward(
    weights: dict[str, jax.Array], cfg: RealConfig, tokens: jax.Array, cache: Cache | None
) -> tuple[jax.Array, Cache]:
    """Run `tokens` (1-D) after the cached ones: logits after the last, and the new cache."""
    seq = tokens.shape[0]
    past = 0 if cache is None else cache[2]
    capacity = -(-(past + seq) // BUCKET) * BUCKET  # the next multiple of BUCKET
    half = cfg.head_dim // 2
    group = cfg.n_heads // cfg.n_kv_heads
    # The new tokens continue where the cache ends: positions past .. past + seq - 1.
    positions = past + jnp.arange(seq)
    inv_freq = cfg.rope_theta ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions.astype(jnp.float32)[:, None] * inv_freq  # (T, half)
    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)
    # The token at position p sees slots 0..p: all cached ones, and the new ones up to
    # itself. Slots after p, filled or not, are hidden.
    visible = jnp.arange(capacity)[None, :] <= positions[:, None]  # (T, capacity)

    def rope(z: jax.Array) -> jax.Array:
        rotated = jnp.concatenate([-z[..., half:], z[..., :half]], axis=-1)  # (a, b) -> (-b, a)
        return z * cos + rotated * sin

    def heads(z: jax.Array, n: int) -> jax.Array:
        return z.reshape(seq, n, cfg.head_dim).transpose(1, 0, 2)  # (n, T, head_dim)

    def slots(old: jax.Array | None) -> jax.Array:
        """This layer's buffer with room for the new tokens."""
        if old is None:
            return jnp.zeros((cfg.n_kv_heads, capacity, cfg.head_dim), dtype=jnp.float32)
        return jnp.pad(old, ((0, 0), (0, capacity - old.shape[1]), (0, 0)))

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier
    keys, values = [], []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        # A key is rotated once, for its own position, and cached rotated.
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        # .at[].set returns new arrays: the cache that was passed in stays as it was.
        k = slots(None if cache is None else cache[0][i]).at[:, positions].set(k)
        v = slots(None if cache is None else cache[1][i]).at[:, positions].set(v)
        keys.append(k)  # at KV-head width
        values.append(v)
        # Each KV head serves `group` consecutive query heads: expanded for use only.
        scores = q @ jnp.repeat(k, group, axis=0).transpose(0, 2, 1)
        scores = jnp.where(visible, scores * cfg.attention_scale, -jnp.inf)
        mixed = ops.softmax(scores) @ jnp.repeat(v, group, axis=0)
        mixed = mixed.transpose(1, 0, 2).reshape(seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate, then up
        hidden = jax.nn.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    # Only the last position chooses a token: one row through the norm and the head.
    last = ops.rms_norm(x[-1], weights["model.norm.weight"], cfg.norm_eps)
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (head @ last) / cfg.logits_scaling, (keys, values, past + seq)


def prefill(
    weights: dict[str, jax.Array], cfg: RealConfig, prompt_ids: jax.Array
) -> tuple[jax.Array, Cache]:
    return _forward(weights, cfg, prompt_ids, None)


def decode_step(
    weights: dict[str, jax.Array], cfg: RealConfig, token: int, cache: Cache
) -> tuple[jax.Array, Cache]:
    return _forward(weights, cfg, jnp.array([token]), cache)


def generate(
    weights: dict[str, jax.Array], cfg: RealConfig, prompt_ids: jax.Array, max_new_tokens: int
) -> list[int]:
    logits, cache = prefill(weights, cfg, prompt_ids)  # the whole prompt, once
    tokens = []
    for step in range(max_new_tokens):
        token = int(jnp.argmax(logits))
        tokens.append(token)
        if step + 1 < max_new_tokens:  # the last token's logits would never be read
            logits, cache = decode_step(weights, cfg, token, cache)  # one token per step
    return tokens
