import mlx.core as mx

from whetstone.kit import ops


def granite_block(
    x: mx.array, weights: dict[str, mx.array], layer: int, positions: mx.array
) -> mx.array:
    """One decoder block of Granite 4.0 (model.layers.{layer}), from checkpoint tensors.

    x: float32, shape (B, T, D): the hidden state entering the block.
    weights: mapping from checkpoint names to float32 tensors, (out, in) for matrices:
        "model.layers.{layer}.input_layernorm.weight"            (D,)
        "model.layers.{layer}.self_attn.q_proj.weight"           (H * 64, D)
        "model.layers.{layer}.self_attn.k_proj.weight"           (H_kv * 64, D)
        "model.layers.{layer}.self_attn.v_proj.weight"           (H_kv * 64, D)
        "model.layers.{layer}.self_attn.o_proj.weight"           (D, H * 64)
        "model.layers.{layer}.post_attention_layernorm.weight"   (D,)
        "model.layers.{layer}.shared_mlp.input_linear.weight"    (2 * F, D): gate rows, then up rows
        "model.layers.{layer}.shared_mlp.output_linear.weight"   (D, F)
      It holds the other layers too. Read H, H_kv, and F from the shapes.
    layer: which block.
    positions: integer, shape (T,): the absolute position of each token, increasing.
    Constants of the model: head width 64, RMSNorm eps 1e-5, RoPE theta 1e7 (half-split
    pairs), attention scores times 1/64, both branches times 0.263 before the residual.
    Building blocks: ops.rms_norm(x, weight, eps), ops.softmax(x).
    Returns float32, shape (B, T, D): the hidden state leaving the block.
    """
    raise NotImplementedError
