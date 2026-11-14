from typing import Any

import torch

from whetstone.kit import RealConfig, ops

# The cache is yours to design: anything that holds, for every layer, the keys and
# values of the tokens read so far. The tests only pass it back to decode_step.
Cache = Any


def prefill(
    weights: dict[str, torch.Tensor], cfg: RealConfig, prompt_ids: torch.Tensor
) -> tuple[torch.Tensor, Cache]:
    """Read the whole prompt in one forward pass.

    weights: mapping from checkpoint names to float32 tensors, (out, in) for matrices:
        "model.embed_tokens.weight"                          (vocab_size, d_model)
        "model.layers.{i}.input_layernorm.weight"            (d_model,)
        "model.layers.{i}.self_attn.q_proj.weight"           (n_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.k_proj.weight"           (n_kv_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.v_proj.weight"           (n_kv_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.o_proj.weight"           (d_model, n_heads * head_dim)
        "model.layers.{i}.post_attention_layernorm.weight"   (d_model,)
        "model.layers.{i}.shared_mlp.input_linear.weight"    (2 * d_ff, d_model): gate, then up
        "model.layers.{i}.shared_mlp.output_linear.weight"   (d_model, d_ff)
        "model.norm.weight"                                  (d_model,)
        "lm_head.weight"                                     (vocab_size, d_model), untied only
    cfg: n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab_size, norm_eps,
        rope_theta, attention_scale, embedding_multiplier, residual_multiplier,
        logits_scaling, tied_embeddings.
    prompt_ids: integer, shape (P,), P >= 1. Token t is at position t.
    Building blocks: ops.rms_norm(x, weight, eps), ops.softmax(x).
    Returns (logits, cache): the float32 logits of the token after the prompt, shape
    (vocab_size,), and a cache that holds the P prompt tokens.
    """
    raise NotImplementedError


def decode_step(
    weights: dict[str, torch.Tensor], cfg: RealConfig, token: int, cache: Cache
) -> tuple[torch.Tensor, Cache]:
    """Read one more token, using the cache for everything before it.

    token: a Python int, the token that follows the ones in `cache`.
    cache: what prefill or decode_step returned. It must not be modified: the caller
        may continue from the same cache again.
    Returns (logits, cache): the float32 logits of the next token, shape (vocab_size,),
    and a new cache that holds one token more.
    """
    raise NotImplementedError


def generate(
    weights: dict[str, torch.Tensor], cfg: RealConfig, prompt_ids: torch.Tensor, max_new_tokens: int
) -> list[int]:
    """Greedy generation: prefill once, then one decode_step per token.

    Returns the max_new_tokens generated token ids (without the prompt) as Python ints.
    """
    raise NotImplementedError
