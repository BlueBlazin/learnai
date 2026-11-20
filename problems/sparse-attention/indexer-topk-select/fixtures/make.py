"""Regenerate the fixture of indexer-topk-select.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/sparse-attention/indexer-topk-select/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
arrays are what ``GlmMoeDsaIndexer`` of ``models/glm_moe_dsa/modeling_glm_moe_dsa.py``
computes, so the tests compare a solution with the shipped code and not only with this
problem's own reference.

The indexer is built with the indexer sizes of ``zai-org/GLM-5.3`` (``config.json`` read
on 2026-11-21: ``index_n_heads`` 32, ``index_head_dim`` 128, ``qk_rope_head_dim`` 64) on
a narrow residual stream (``hidden_size`` 48, ``q_lora_rank`` 24) with random weights.
``index_topk`` is 2048 in the shipped config and 4 here, so that 9 tokens exercise it.

What is saved, for 9 tokens at positions 0 to 8:

- ``q_idx`` ``(9, 32, 128)`` and ``k_idx`` ``(9, 128)``: the two tensors the indexer
  multiplies, after its rotary embedding, computed with the module's own layers by the
  lines of ``GlmMoeDsaIndexer.forward``;
- ``w`` ``(9, 32)``: ``weights_proj(hidden_states)`` before the ``H ** -0.5`` scale;
- ``scores`` ``(9, 9)``: the tensor the module's forward hands to ``topk`` (captured
  there), ``-inf`` above the diagonal;
- ``visible`` ``(9, 9)``: the causal mask given to the module;
- ``k`` and ``mask`` ``(9, 9)``: the indices the forward returns for ``index_topk = k``,
  scattered into a mask and intersected with ``visible``, which is what
  ``GlmMoeDsaAttention`` does with them.

The script asserts that no row has two equal scores at the cut, so ``mask`` does not
depend on the order ``torch.topk`` gives equal scores.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import torch
import transformers
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as glm
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

HERE = Path(__file__).parent
STEPS, K = 9, 4
HEADS, DIM, ROPE_DIM = 32, 128, 64  # GLM-5.3: index_n_heads, index_head_dim, qk_rope_head_dim


def main() -> None:
    torch.manual_seed(2201)
    config = GlmMoeDsaConfig(
        vocab_size=16,
        hidden_size=48,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=8,
        q_lora_rank=24,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=4,
        v_head_dim=8,
        index_topk=K,
        index_head_dim=DIM,
        index_n_heads=HEADS,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    indexer = glm.GlmMoeDsaIndexer(config, 0).eval()
    with torch.no_grad():  # a LayerNorm away from its initial weight 1 and bias 0
        indexer.k_norm.weight.uniform_(0.5, 1.5)
        indexer.k_norm.bias.normal_(0.0, 0.1)
        # Default initialization gives head weights of about 0.2; make them of order 1.
        indexer.weights_proj.weight.mul_(4.0)
    rotary = glm.GlmMoeDsaRotaryEmbedding(config)

    hidden = torch.randn(1, STEPS, config.hidden_size)
    q_resid = torch.randn(1, STEPS, config.q_lora_rank)
    positions = torch.arange(STEPS)[None]
    cos, sin = rotary(hidden, positions)
    visible = (torch.arange(STEPS)[None, :] <= torch.arange(STEPS)[:, None])[None]

    # The module's forward, with the tensor it hands to topk captured.
    seen = {}
    real_topk = torch.Tensor.topk

    def spy(self, *args, **kwargs):
        seen["scores"] = self.detach().clone()
        return real_topk(self, *args, **kwargs)

    with mock.patch.object(torch.Tensor, "topk", spy), torch.no_grad():
        indices = indexer(hidden, q_resid, (cos, sin), visible, positions)
    scores = seen["scores"][0]
    assert indices.shape == (1, STEPS, K)
    mask = torch.zeros(STEPS, STEPS, dtype=torch.bool).scatter(-1, indices[0].long(), True)
    mask &= visible[0]

    # The operands of its dot product, by the lines of GlmMoeDsaIndexer.forward.
    with torch.no_grad():
        q = indexer.wq_b(q_resid).view(1, STEPS, HEADS, DIM)
        q_rot, q_pass = torch.split(q, [ROPE_DIM, DIM - ROPE_DIM], dim=-1)
        k = indexer.k_norm(indexer.wk(hidden)).unsqueeze(2)
        k_rot, k_pass = torch.split(k, [ROPE_DIM, DIM - ROPE_DIM], dim=-1)
        q_rot, k_rot = glm.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q_idx = torch.cat([q_rot, q_pass], dim=-1)[0]
        k_idx = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)[0]
        w = indexer.weights_proj(hidden)[0]

    # No ties at the cut: the mask does not depend on torch.topk's order of equal scores.
    ranked = torch.sort(scores, dim=-1, descending=True).values
    for t in range(K, STEPS):
        assert ranked[t, K - 1] - ranked[t, K] > 1e-3, (t, ranked[t])
    assert (scores[visible[0]] < 0).any() and (scores[visible[0]] > 0).any()

    out = {
        "transformers_version": np.array(transformers.__version__),
        "q_idx": q_idx.numpy(),
        "k_idx": k_idx.numpy(),
        "w": w.numpy(),
        "scores": scores.numpy(),
        "visible": visible[0].numpy(),
        "k": np.array(K),
        "mask": mask.numpy(),
    }
    np.savez_compressed(HERE / "shipped.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)
    print("visible scores from", float(scores[visible[0]].min()), "to", float(scores.max()))


if __name__ == "__main__":
    main()
