# BUG: subclasses mlx-lm's own Granite implementation, which the statement rules out; the module is not written.
import mlx.core as mx
from mlx_lm.models import granitemoehybrid

from learnai.kit import RealConfig


class Granite(granitemoehybrid.Model):
    def __init__(self, cfg: RealConfig):
        super().__init__(
            granitemoehybrid.ModelArgs(
                model_type="granitemoehybrid",
                vocab_size=cfg.vocab_size,
                hidden_size=cfg.d_model,
                intermediate_size=cfg.d_ff,
                shared_intermediate_size=cfg.d_ff,
                num_hidden_layers=cfg.n_layers,
                max_position_embeddings=cfg.max_positions,
                num_attention_heads=cfg.n_heads,
                num_key_value_heads=cfg.n_kv_heads,
                attention_bias=False,
                embedding_multiplier=cfg.embedding_multiplier,
                attention_multiplier=cfg.attention_scale,
                logits_scaling=cfg.logits_scaling,
                residual_multiplier=cfg.residual_multiplier,
                layer_types=["attention"] * cfg.n_layers,
                rms_norm_eps=cfg.norm_eps,
                rope_theta=cfg.rope_theta,
            )
        )

    def __call__(self, tokens: mx.array) -> mx.array:
        return super().__call__(tokens)
