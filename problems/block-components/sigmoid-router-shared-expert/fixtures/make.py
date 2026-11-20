"""Regenerate the fixture of sigmoid-router-shared-expert.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/sigmoid-router-shared-expert/fixtures/make.py

Needs the ``models`` extra (``transformers``); no weights are downloaded. The expected
routing weights are the outputs of two router modules of ``transformers``, given random
router weights, a random selection bias, and random token vectors:

- ``glm_*``: ``GlmMoeDsaTopkRouter`` (``models/glm_moe_dsa/modeling_glm_moe_dsa.py``)
  with the routing parameters of ``zai-org/GLM-5.3`` (``config.json`` read on
  2026-11-21): 256 routed experts, 8 per token, ``routed_scaling_factor`` 2.5, sigmoid
  scores, one expert group, ``norm_topk_prob`` true.
- ``ds_*``: ``DeepseekV4TopKRouter`` (``models/deepseek_v4/modeling_deepseek_v4.py``)
  with those of ``deepseek-ai/DeepSeek-V4-Pro-0813`` (same date): 384 routed experts,
  6 per token, ``routed_scaling_factor`` 2.5, ``scoring_func`` "sqrtsoftplus".

Only the hidden width is reduced (16 instead of 6144 and 7168). The library returns the
chosen experts in no particular order, so the fixture stores the weights as a dense
``(tokens, experts)`` matrix, zero for the experts a token did not choose. Inputs are
redrawn until every token has a clear margin at the top-k boundary, so float rounding
cannot change a selection.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaTopkRouter

HERE = Path(__file__).parent
HIDDEN, TOKENS = 16, 6
MARGIN = 1e-3


def run(router, rng, n_experts, top_k):
    """Random parameters and inputs with a clear top-k margin; the dense weights."""
    while True:
        w = (rng.standard_normal((n_experts, HIDDEN)) * 0.35).astype(np.float32)
        bias = (rng.standard_normal(n_experts) * 0.1).astype(np.float32)
        x = rng.standard_normal((TOKENS, HIDDEN)).astype(np.float32)
        with torch.no_grad():
            router.weight.copy_(torch.from_numpy(w))
            router.e_score_correction_bias.copy_(torch.from_numpy(bias))
            logits, weights, indices = router(torch.from_numpy(x)[None])
            scores = router.score_fn(logits) if hasattr(router, "score_fn") else logits.sigmoid()
            ranked = torch.sort(scores + router.e_score_correction_bias, descending=True).values
        if float((ranked[:, top_k - 1] - ranked[:, top_k]).min()) > MARGIN:
            dense = torch.zeros(TOKENS, n_experts).scatter(1, indices, weights)
            return {"x": x, "w": w, "bias": bias, "dense": dense.numpy()}


def main() -> None:
    rng = np.random.default_rng(2304)
    out = {"transformers_version": np.array(transformers.__version__)}

    glm = SimpleNamespace(
        num_experts_per_tok=8,
        num_local_experts=256,
        hidden_size=HIDDEN,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
    for key, value in run(GlmMoeDsaTopkRouter(glm), rng, 256, 8).items():
        out[f"glm_{key}"] = value
    out["glm_k"], out["glm_scale"] = np.array(8), np.array(2.5)

    deepseek = SimpleNamespace(
        num_experts_per_tok=6,
        num_local_experts=384,
        hidden_size=HIDDEN,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
    )
    for key, value in run(DeepseekV4TopKRouter(deepseek), rng, 384, 6).items():
        out[f"ds_{key}"] = value
    out["ds_k"], out["ds_scale"] = np.array(6), np.array(2.5)

    np.savez_compressed(HERE / "routers.npz", **out)
    for key, value in out.items():
        print(key, value.shape, value.dtype)


if __name__ == "__main__":
    main()
