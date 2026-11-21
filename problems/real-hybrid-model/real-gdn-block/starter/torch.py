import torch

from learnai.kit import ops


def qwen_gdn_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One linear-attention block of Qwen3.5 (layers.{layer}) on T new rows, from checkpoint tensors.

    x: float32, shape (T, D): the hidden states entering the block.
    weights: mapping from checkpoint names to float32 tensors, (out, in) for matrices.
      With p = "model.language_model.layers.{layer}." the block reads
        p + "input_layernorm.weight", p + "post_attention_layernorm.weight"   (D,)  stored as gain - 1
        p + "linear_attn.in_proj_qkv.weight"   (2 * H * dk + H * dv, D)   q, k, v in that order
        p + "linear_attn.conv1d.weight"        (2 * H * dk + H * dv, 1, K)
        p + "linear_attn.in_proj_a.weight", p + "linear_attn.in_proj_b.weight"   (H, D)
        p + "linear_attn.A_log", p + "linear_attn.dt_bias"   (H,)
        p + "linear_attn.in_proj_z.weight"     (H * dv, D)
        p + "linear_attn.norm.weight"          (dv,)   a plain gain
        p + "linear_attn.out_proj.weight"      (D, H * dv)
        p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"   (F, D)
        p + "mlp.down_proj.weight"             (D, F)
      It holds the other layers too. Read H, dk, dv, K, and F from the shapes; there are
      as many key heads as value heads.
    layer: which block, an int.
    state: None for a new sequence, or (conv_state, S) as returned by the call that read
      the rows before x: conv_state (K - 1, 2 * H * dk + H * dv), S (H, dv, dk).
    Building blocks: ops.rms_norm(x, weight, eps), ops.swiglu(x, w_gate, w_up, w_down),
      ops.l2norm(x), ops.softplus(x), ops.causal_conv1d(x, w, state).
    Returns (x, (conv_state, S)): the hidden states leaving the block, float32 (T, D),
      and the new state. Every RMSNorm epsilon is 1e-6.
    """
    raise NotImplementedError
