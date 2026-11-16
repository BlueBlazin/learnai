import jax
import jax.numpy as jnp

from whetstone.kit import DecoderConfig, ops

Cache = list[tuple[jax.Array, jax.Array]]

EIGHT_BIT = ("model.embed_tokens.weight", "lm_head.weight")  # stage 4: the sensitive matrices


# -- stage 1: quantize and pack -----------------------------------------------------------


def _quantize_groups(w: jax.Array, bits: int, group_size: int):
    """`groupwise-4bit`: one scale and one bias per run of group_size inputs of a row."""
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = jnp.min(groups, axis=-1)
    hi = jnp.max(groups, axis=-1)
    # A constant group gets scale 1: its codes are 0 / 1 = 0 and the bias reproduces it.
    # Divide by an array: JAX turns division by a Python scalar into a multiplication by
    # its reciprocal, which is one bit off for many values.
    scale = jnp.where(hi > lo, (hi - lo) / jnp.full_like(hi, levels), jnp.ones_like(hi))
    codes = jnp.round((groups - lo[..., None]) / scale[..., None])
    codes = jnp.clip(codes, 0, levels).astype(jnp.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo


def _pack(codes: jax.Array, bits: int) -> jax.Array:
    """8-bit codes are bytes already; two 4-bit codes share a byte, the first one low."""
    if bits == 8:
        return codes
    pairs = codes.reshape(codes.shape[0], -1, 2)
    return pairs[..., 0] + 16 * pairs[..., 1]  # at most 15 + 240: still a uint8


def _unpack(packed: jax.Array, bits: int) -> jax.Array:
    if bits == 8:
        return packed
    low = packed % 16
    high = packed // 16
    return jnp.stack((low, high), axis=-1).reshape(*packed.shape[:-1], -1)


def _quantize(params: dict[str, jax.Array], bits_of, group_size: int) -> dict[str, jax.Array]:
    ckpt = {}
    for name, w in params.items():
        if w.ndim != 2:
            ckpt[name] = w  # norm gains stay float32
            continue
        bits = bits_of(name)
        codes, scale, bias = _quantize_groups(w, bits, group_size)
        prefix = name[: -len("weight")]
        # The float matrix is not stored: the codes replace it under its own name.
        ckpt[name] = _pack(codes, bits)
        ckpt[prefix + "scales"] = scale
        ckpt[prefix + "biases"] = bias
    return ckpt


def quantize_checkpoint(
    params: dict[str, jax.Array], bits: int, group_size: int
) -> dict[str, jax.Array]:
    return _quantize(params, lambda name: bits, group_size)


# -- stage 2: the forward pass on codes ---------------------------------------------------


def _matrix(ckpt: dict[str, jax.Array], name: str, group_size: int, rows=None):
    """Codes (as float32), scales, and biases of one matrix, or of some of its rows."""
    prefix = name[: -len("weight")]
    packed, scale, bias = ckpt[name], ckpt[prefix + "scales"], ckpt[prefix + "biases"]
    # The width of a row of bytes tells how many bits a code has.
    bits = 8 * packed.shape[-1] // (scale.shape[-1] * group_size)
    if rows is not None:
        packed, scale, bias = packed[rows], scale[rows], bias[rows]
    return _unpack(packed, bits).astype(jnp.float32), scale, bias


def _qlinear(ckpt: dict[str, jax.Array], name: str, x: jax.Array, group_size: int):
    """`quantized-matmul`: x @ W_hat.T from the codes, never building W_hat."""
    codes, scale, bias = _matrix(ckpt, name, group_size)
    out_dim, n_groups = scale.shape
    xg = x.reshape(*x.shape[:-1], n_groups, group_size)  # (..., G, g)
    cg = codes.reshape(out_dim, n_groups, group_size)  # (O, G, g)
    dots = jnp.einsum("...gk,ogk->...og", xg, cg)  # inputs times raw codes, per group
    sums = xg.sum(axis=-1)  # each bias multiplies the sum of its group's inputs
    return (dots * scale).sum(axis=-1) + sums @ bias.T


def _embed(ckpt: dict[str, jax.Array], tokens: jax.Array, group_size: int):
    """Dequantize only the rows that are looked up: (B, T) ids -> (B, T, d_model)."""
    codes, scale, bias = _matrix(ckpt, "model.embed_tokens.weight", group_size, rows=tokens)
    groups = codes.reshape(*tokens.shape, -1, group_size)
    return (groups * scale[..., None] + bias[..., None]).reshape(*tokens.shape, -1)


# -- stage 3: the cached step ------------------------------------------------------------


def step_q(
    ckpt: dict[str, jax.Array],
    cfg: DecoderConfig,
    tokens: jax.Array,
    cache: Cache | None,
    group_size: int,
) -> tuple[jax.Array, Cache]:
    batch, new = tokens.shape
    past = 0 if cache is None else cache[0][0].shape[2]
    # New token i sits at absolute position past + i: that is its RoPE position, and it
    # may attend to every cached entry and to the new entries up to its own.
    positions = past + jnp.arange(new)
    visible = jnp.arange(past + new)[None, :] <= positions[:, None]  # (T, S + T)

    def heads(t: jax.Array, n: int) -> jax.Array:
        return t.reshape(batch, new, n, cfg.head_dim).transpose(0, 2, 1, 3)  # (B, n, T, head_dim)

    def linear(name: str, t: jax.Array) -> jax.Array:
        return _qlinear(ckpt, name, t, group_size)

    x = _embed(ckpt, tokens, group_size)
    new_cache = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, ckpt[p + "input_layernorm.weight"], cfg.norm_eps)
        q = heads(linear(p + "self_attn.q_proj.weight", h), cfg.n_heads)
        k = heads(linear(p + "self_attn.k_proj.weight", h), cfg.n_kv_heads)
        v = heads(linear(p + "self_attn.v_proj.weight", h), cfg.n_kv_heads)
        q = ops.rope(q, positions, cfg.rope_base)
        k = ops.rope(k, positions, cfg.rope_base)  # keys are cached rotated
        if cache is not None:
            k = jnp.concatenate((cache[i][0], k), axis=2)
            v = jnp.concatenate((cache[i][1], v), axis=2)
        new_cache.append((k, v))
        attended = ops.attention(q, k, v, visible)  # (B, n_heads, T, head_dim)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, new, cfg.d_model)
        x = x + linear(p + "self_attn.o_proj.weight", attended)
        h = ops.rms_norm(x, ckpt[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        gate = linear(p + "mlp.gate_proj.weight", h)
        up = linear(p + "mlp.up_proj.weight", h)
        x = x + linear(p + "mlp.down_proj.weight", gate * jax.nn.sigmoid(gate) * up)  # SwiGLU
    x = ops.rms_norm(x, ckpt["model.norm.weight"], cfg.norm_eps)
    # A tied checkpoint has no lm_head: the head multiplies by the embedding's codes.
    head = "model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"
    return linear(head, x), new_cache


def forward_q(
    ckpt: dict[str, jax.Array], cfg: DecoderConfig, tokens: jax.Array, group_size: int
) -> jax.Array:
    # The uncached pass is the cached step with nothing cached.
    return step_q(ckpt, cfg, tokens, None, group_size)[0]


def generate_q(
    ckpt: dict[str, jax.Array],
    cfg: DecoderConfig,
    prompt: list[int],
    max_new_tokens: int,
    group_size: int,
    cache: Cache | None = None,
) -> list[int]:
    # The prompt goes through once; after that every step feeds one token.
    feed = jnp.array([prompt])
    generated: list[int] = []
    while True:
        logits, cache = step_q(ckpt, cfg, feed, cache, group_size)
        generated.append(int(jnp.argmax(logits[0, -1])))
        if len(generated) == max_new_tokens:
            return generated
        feed = jnp.array([[generated[-1]]])


# -- stage 4: the mixed recipe -----------------------------------------------------------


def bits_per_weight(ckpt: dict[str, jax.Array], group_size: int) -> float:
    stored_bits = parameters = 0
    for name, tensor in ckpt.items():
        stored_bits += tensor.size * (8 if tensor.dtype == jnp.uint8 else 32)
        if name.endswith(".scales"):
            parameters += tensor.size * group_size  # one group stands for group_size weights
        elif tensor.dtype != jnp.uint8 and not name.endswith(".biases"):
            parameters += tensor.size  # a tensor that was not quantized
    return stored_bits / parameters


def quantize_mixed(
    params: dict[str, jax.Array], group_size: int, max_bpw: float
) -> dict[str, jax.Array]:
    ckpt = _quantize(params, lambda name: 8 if name in EIGHT_BIT else 4, group_size)
    if bits_per_weight(ckpt, group_size) > max_bpw:
        raise ValueError("the mixed recipe does not fit the budget")
    return ckpt
