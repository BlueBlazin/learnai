import mlx.core as mx

from learnai.kit import RealConfig
from mlx import nn


class Granite(nn.Module):
    """IBM Granite 4.0 (dense, attention-only layers) as an mlx.nn module tree.

    The parameter names of the tree, as mlx.utils.tree_flatten spells them, are exactly
    the tensor names of the checkpoint, so `model.load_weights(path)` works unchanged:
    model.embed_tokens.weight, model.layers.{i}.input_layernorm.weight,
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight,
    model.layers.{i}.post_attention_layernorm.weight,
    model.layers.{i}.shared_mlp.{input,output}_linear.weight, model.norm.weight, and
    lm_head.weight only when cfg.tied_embeddings is False.
    """

    def __init__(self, cfg: RealConfig):
        """cfg: n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab_size,
        norm_eps, rope_theta, attention_scale, embedding_multiplier,
        residual_multiplier, logits_scaling, tied_embeddings."""
        super().__init__()
        raise NotImplementedError

    def __call__(self, tokens: mx.array) -> mx.array:
        """tokens: integer array (B, T). Returns the logits, (B, T, vocab_size), in
        the dtype of the parameters, computed with mx.fast.scaled_dot_product_attention,
        RoPE (nn.RoPE or mx.fast.rope), and RMSNorm (nn.RMSNorm or mx.fast.rms_norm)."""
        raise NotImplementedError
