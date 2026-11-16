import jax

from learnai.kit import ops


def block(x: jax.Array, params: dict[str, jax.Array], mask: jax.Array) -> jax.Array:
    """One pre-norm decoder block with single-head attention.

    x: float32, shape (B, T, D). The residual stream entering the block.
    params: the block's weights, keyed by checkpoint-style names (linear weights are (out, in)):
        "input_layernorm.weight"            (D,)    gain of the norm before attention
        "self_attn.q_proj.weight"           (D, D)
        "self_attn.k_proj.weight"           (D, D)
        "self_attn.v_proj.weight"           (D, D)
        "self_attn.o_proj.weight"           (D, D)
        "post_attention_layernorm.weight"   (D,)    gain of the norm before the MLP
        "mlp.gate_proj.weight"              (M, D)
        "mlp.up_proj.weight"                (M, D)
        "mlp.down_proj.weight"              (D, M)
    mask: bool, shape (T, T). True where query position i may attend to key position j.
    Building blocks: ops.rms_norm(x, weight), ops.attention(q, k, v, mask),
    ops.swiglu(x, w_gate, w_up, w_down).
    Returns float32, shape (B, T, D).
    """
    raise NotImplementedError
