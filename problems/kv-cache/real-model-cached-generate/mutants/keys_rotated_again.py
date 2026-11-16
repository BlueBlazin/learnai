# BUG: applies RoPE to all keys after appending, so every cached key, already rotated when it was new, is rotated again at each step.
import torch

from learnai.kit import RealConfig, ops

# The cache: for each layer, (keys, values) of every token read so far, each of shape
# (n_kv_heads, length, head_dim). Keys are stored already rotated.
Cache = list[tuple[torch.Tensor, torch.Tensor]]


def _forward(
    weights: dict[str, torch.Tensor], cfg: RealConfig, tokens: torch.Tensor, cache: Cache | None
) -> tuple[torch.Tensor, Cache]:
    """Run `tokens` (1-D) after the cached ones: logits after the last, and the new cache."""
    seq = tokens.shape[0]
    past = 0 if cache is None else cache[0][0].shape[1]
    half = cfg.head_dim // 2
    group = cfg.n_heads // cfg.n_kv_heads
    # The new tokens continue where the cache ends: positions past .. past + seq - 1.
    positions = past + torch.arange(seq)
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq  # (T, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    # The token at position p sees entries 0..p: all cached ones, and the new ones up
    # to itself. For a single new token every entry is visible.
    visible = torch.arange(past + seq)[None, :] <= positions[:, None]  # (T, past + T)

    def rope(z: torch.Tensor) -> torch.Tensor:
        rotated = torch.cat([-z[..., half:], z[..., :half]], dim=-1)  # (a, b) -> (-b, a)
        return z * cos + rotated * sin

    def rope_all(z: torch.Tensor) -> torch.Tensor:
        every = torch.arange(z.shape[1], dtype=torch.float32)[:, None] * inv_freq
        cos_all = torch.cat([every.cos(), every.cos()], dim=-1)
        sin_all = torch.cat([every.sin(), every.sin()], dim=-1)
        rotated = torch.cat([-z[..., half:], z[..., :half]], dim=-1)
        return z * cos_all + rotated * sin_all

    def heads(z: torch.Tensor, n: int) -> torch.Tensor:
        return z.reshape(seq, n, cfg.head_dim).transpose(0, 1)  # (n, T, head_dim)

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier
    new_cache = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        k = heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads)
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        if cache is not None:
            # cat builds new tensors: the cache that was passed in stays as it was.
            k = torch.cat([cache[i][0], k], dim=1)
            v = torch.cat([cache[i][1], v], dim=1)
        k = rope_all(k)  # the cached part was rotated in an earlier call already
        new_cache.append((k, v))  # at KV-head width
        # Each KV head serves `group` consecutive query heads: expanded for use only.
        scores = q @ k.repeat_interleave(group, dim=0).transpose(-1, -2)
        scores = (scores * cfg.attention_scale).masked_fill(~visible, float("-inf"))
        mixed = ops.softmax(scores) @ v.repeat_interleave(group, dim=0)
        mixed = mixed.transpose(0, 1).reshape(seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate, then up
        hidden = torch.nn.functional.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    # Only the last position chooses a token: one row through the norm and the head.
    last = ops.rms_norm(x[-1], weights["model.norm.weight"], cfg.norm_eps)
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (head @ last) / cfg.logits_scaling, new_cache


def prefill(
    weights: dict[str, torch.Tensor], cfg: RealConfig, prompt_ids: torch.Tensor
) -> tuple[torch.Tensor, Cache]:
    return _forward(weights, cfg, prompt_ids, None)


def decode_step(
    weights: dict[str, torch.Tensor], cfg: RealConfig, token: int, cache: Cache
) -> tuple[torch.Tensor, Cache]:
    return _forward(weights, cfg, torch.tensor([token]), cache)


def generate(
    weights: dict[str, torch.Tensor], cfg: RealConfig, prompt_ids: torch.Tensor, max_new_tokens: int
) -> list[int]:
    logits, cache = prefill(weights, cfg, prompt_ids)  # the whole prompt, once
    tokens = []
    for step in range(max_new_tokens):
        token = int(logits.argmax())
        tokens.append(token)
        if step + 1 < max_new_tokens:  # the last token's logits would never be read
            logits, cache = decode_step(weights, cfg, token, cache)  # one token per step
    return tokens
