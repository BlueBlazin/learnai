import jax
import jax.numpy as jnp

from learnai.kit import HybridConfig, ops


def layer_types(n_layers: int, interval: int) -> list[str]:
    """The mixer of each layer: a list of `n_layers` strings, each "linear_attention" or
    "full_attention"."""
    raise NotImplementedError


def hybrid_forward(
    params: dict[str, jax.Array],
    cfg: HybridConfig,
    tokens: jax.Array,
    state: list[tuple[jax.Array, jax.Array]] | None,
) -> tuple[jax.Array, list[tuple[jax.Array, jax.Array]]]:
    """The hybrid decoder on T new tokens, continuing from `state`.

    params: the checkpoint's tensors by name, float32, linear weights (out, in):
        "model.embed_tokens.weight"                      (vocab_size, d_model), also the head
        "model.layers.{i}.input_layernorm.weight"        (d_model,)
        "model.layers.{i}.post_attention_layernorm.weight"   (d_model,)
        "model.layers.{i}.mlp.{gate,up}_proj.weight"     (d_mlp, d_model)
        "model.layers.{i}.mlp.down_proj.weight"          (d_model, d_mlp)
        "model.norm.weight"                              (d_model,)
      a linear_attention layer has the nine tensors of `gated-delta-layer` under
        "model.layers.{i}.linear_attn."
      a full_attention layer has, under "model.layers.{i}.self_attn.",
        "q_proj.weight"                  (n_heads * 2 * head_dim, d_model)
        "k_proj.weight", "v_proj.weight" (n_kv_heads * head_dim, d_model)
        "o_proj.weight"                  (d_model, n_heads * head_dim)
        "q_norm.weight", "k_norm.weight" (head_dim,)
    cfg: a HybridConfig. cfg.layers[i] is "linear_attention" or "full_attention".
    tokens: integer, shape (T,), T >= 1.
    state: None for a new sequence, or the list an earlier call returned: one pair per
        layer, (conv_state, S) for a linear_attention layer and (k, v) for a
        full_attention layer, with k and v of shape (n_kv_heads, tokens so far, head_dim).
    Returns (logits, state): logits float32 of shape (T, vocab_size), and the new list.
    Nothing passed in is modified.
    """
    raise NotImplementedError
