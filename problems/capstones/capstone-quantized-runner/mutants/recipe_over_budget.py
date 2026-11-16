# BUG: quantize_mixed never compares the size of its checkpoint with max_bpw.
import torch

from whetstone.kit import DecoderConfig, ops

Cache = list[tuple[torch.Tensor, torch.Tensor]]

EIGHT_BIT = ("model.embed_tokens.weight", "lm_head.weight")  # stage 4: the sensitive matrices


# -- stage 1: quantize and pack -----------------------------------------------------------


def _quantize_groups(w: torch.Tensor, bits: int, group_size: int):
    """`groupwise-4bit`: one scale and one bias per run of group_size inputs of a row."""
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    # A constant group gets scale 1: its codes are 0 / 1 = 0 and the bias reproduces it.
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """8-bit codes are bytes already; two 4-bit codes share a byte, the first one low."""
    if bits == 8:
        return codes
    pairs = codes.reshape(codes.shape[0], -1, 2)
    return pairs[..., 0] + 16 * pairs[..., 1]  # at most 15 + 240: still a uint8


def _unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return packed
    low = packed % 16
    high = packed // 16
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)


def _quantize(params: dict[str, torch.Tensor], bits_of, group_size: int) -> dict[str, torch.Tensor]:
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
    params: dict[str, torch.Tensor], bits: int, group_size: int
) -> dict[str, torch.Tensor]:
    return _quantize(params, lambda name: bits, group_size)


# -- stage 2: the forward pass on codes ---------------------------------------------------


def _matrix(ckpt: dict[str, torch.Tensor], name: str, group_size: int, rows=None):
    """Codes (as float32), scales, and biases of one matrix, or of some of its rows."""
    prefix = name[: -len("weight")]
    packed, scale, bias = ckpt[name], ckpt[prefix + "scales"], ckpt[prefix + "biases"]
    # The width of a row of bytes tells how many bits a code has.
    bits = 8 * packed.shape[-1] // (scale.shape[-1] * group_size)
    if rows is not None:
        packed, scale, bias = packed[rows], scale[rows], bias[rows]
    return _unpack(packed, bits).to(torch.float32), scale, bias


def _qlinear(ckpt: dict[str, torch.Tensor], name: str, x: torch.Tensor, group_size: int):
    """`quantized-matmul`: x @ W_hat.T from the codes, never building W_hat."""
    codes, scale, bias = _matrix(ckpt, name, group_size)
    out_dim, n_groups = scale.shape
    xg = x.reshape(*x.shape[:-1], n_groups, group_size)  # (..., G, g)
    cg = codes.reshape(out_dim, n_groups, group_size)  # (O, G, g)
    dots = torch.einsum("...gk,ogk->...og", xg, cg)  # inputs times raw codes, per group
    sums = xg.sum(dim=-1)  # each bias multiplies the sum of its group's inputs
    return (dots * scale).sum(dim=-1) + sums @ bias.T


def _embed(ckpt: dict[str, torch.Tensor], tokens: torch.Tensor, group_size: int):
    """Dequantize only the rows that are looked up: (B, T) ids -> (B, T, d_model)."""
    codes, scale, bias = _matrix(ckpt, "model.embed_tokens.weight", group_size, rows=tokens)
    groups = codes.reshape(*tokens.shape, -1, group_size)
    return (groups * scale[..., None] + bias[..., None]).reshape(*tokens.shape, -1)


# -- stage 3: the cached step ------------------------------------------------------------


def step_q(
    ckpt: dict[str, torch.Tensor],
    cfg: DecoderConfig,
    tokens: torch.Tensor,
    cache: Cache | None,
    group_size: int,
) -> tuple[torch.Tensor, Cache]:
    batch, new = tokens.shape
    past = 0 if cache is None else cache[0][0].shape[2]
    # New token i sits at absolute position past + i: that is its RoPE position, and it
    # may attend to every cached entry and to the new entries up to its own.
    positions = past + torch.arange(new)
    visible = torch.arange(past + new)[None, :] <= positions[:, None]  # (T, S + T)

    def heads(t: torch.Tensor, n: int) -> torch.Tensor:
        return t.reshape(batch, new, n, cfg.head_dim).transpose(1, 2)  # (B, n, T, head_dim)

    def linear(name: str, t: torch.Tensor) -> torch.Tensor:
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
            k = torch.cat((cache[i][0], k), dim=2)
            v = torch.cat((cache[i][1], v), dim=2)
        new_cache.append((k, v))
        attended = ops.attention(q, k, v, visible)  # (B, n_heads, T, head_dim)
        attended = attended.transpose(1, 2).reshape(batch, new, cfg.d_model)
        x = x + linear(p + "self_attn.o_proj.weight", attended)
        h = ops.rms_norm(x, ckpt[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        gate = linear(p + "mlp.gate_proj.weight", h)
        up = linear(p + "mlp.up_proj.weight", h)
        x = x + linear(p + "mlp.down_proj.weight", gate * torch.sigmoid(gate) * up)  # SwiGLU
    x = ops.rms_norm(x, ckpt["model.norm.weight"], cfg.norm_eps)
    # A tied checkpoint has no lm_head: the head multiplies by the embedding's codes.
    head = "model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"
    return linear(head, x), new_cache


def forward_q(
    ckpt: dict[str, torch.Tensor], cfg: DecoderConfig, tokens: torch.Tensor, group_size: int
) -> torch.Tensor:
    # The uncached pass is the cached step with nothing cached.
    return step_q(ckpt, cfg, tokens, None, group_size)[0]


def generate_q(
    ckpt: dict[str, torch.Tensor],
    cfg: DecoderConfig,
    prompt: list[int],
    max_new_tokens: int,
    group_size: int,
    cache: Cache | None = None,
) -> list[int]:
    # The prompt goes through once; after that every step feeds one token.
    feed = torch.tensor([prompt])
    generated: list[int] = []
    while True:
        logits, cache = step_q(ckpt, cfg, feed, cache, group_size)
        generated.append(int(torch.argmax(logits[0, -1])))
        if len(generated) == max_new_tokens:
            return generated
        feed = torch.tensor([[generated[-1]]])


# -- stage 4: the mixed recipe -----------------------------------------------------------


def bits_per_weight(ckpt: dict[str, torch.Tensor], group_size: int) -> float:
    stored_bits = parameters = 0
    for name, tensor in ckpt.items():
        stored_bits += tensor.numel() * (8 if tensor.dtype == torch.uint8 else 32)
        if name.endswith(".scales"):
            parameters += tensor.numel() * group_size  # one group stands for group_size weights
        elif tensor.dtype != torch.uint8 and not name.endswith(".biases"):
            parameters += tensor.numel()  # a tensor that was not quantized
    return stored_bits / parameters


def quantize_mixed(
    params: dict[str, torch.Tensor], group_size: int, max_bpw: float
) -> dict[str, torch.Tensor]:
    ckpt = _quantize(params, lambda name: 8 if name in EIGHT_BIT else 4, group_size)
    return ckpt
