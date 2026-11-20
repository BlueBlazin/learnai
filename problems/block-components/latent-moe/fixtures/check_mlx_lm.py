"""Check this problem's order of operations against mlx-lm. Writes no fixture.

    uv run --project python --extra jax --extra mlx --extra models \
        python problems/block-components/latent-moe/fixtures/check_mlx_lm.py

Builds ``KimiK3SparseMoE`` of ``mlx_lm/models/kimi_k3.py`` (mlx-lm 0.32.2) at a small
size, gives it random parameters, and compares its output with the formulas of the
statement written in numpy: route at full width, project down once, mix the chosen
experts at latent width, one RMS norm on the mixture, project up, add the full-width
shared expert. It also shows that the two misplacements of the norm, and the other
common epsilon, give different numbers.

Every setting that is not a size is the one in the ``config.json`` of
``moonshotai/Kimi-K3.1-Pro`` (read on 2026-11-21): ``rms_norm_eps`` 1e-5 (the default
of ``TextArgs``, not overridden here), ``latent_moe_use_norm`` true, a sigmoid router
with ``moe_renormalize`` true and ``routed_scaling_factor`` 1.0, one expert group, 2
shared experts, ``activation_situ_beta`` 4.0 and ``activation_situ_linear_beta`` 25.0.

Two things differ from the problem and are reproduced here rather than in the tests:
the library's experts use its SiTU activation, ``beta * tanh(gate / beta) *
sigmoid(gate)`` times ``linear_beta * tanh(up / linear_beta)``, where the kit's
``CountingExperts`` are SwiGLU; and the library adds a selection bias to the scores,
which this problem leaves out. That is why the result is a check of the reading and
not a fixture. Run on 2026-11-21: the statement's order agrees to 1e-6 on outputs of
magnitude 5.
"""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.kimi_k3 import KimiK3SparseMoE, TextArgs

TOKENS, K = 5, 3


BETA, LINEAR_BETA, EPS = 4.0, 25.0, 1e-5


def situ(up: np.ndarray, gate: np.ndarray) -> np.ndarray:
    gated = BETA * np.tanh(gate / BETA) / (1.0 + np.exp(-gate))
    return gated * LINEAR_BETA * np.tanh(up / LINEAR_BETA)


def main() -> None:
    args = TextArgs(
        hidden_size=16,
        num_experts=8,
        num_experts_per_token=K,
        num_shared_experts=2,
        moe_intermediate_size=12,
        routed_expert_hidden_size=8,
        latent_moe_use_norm=True,
        activation_situ_beta=BETA,
        activation_situ_linear_beta=LINEAR_BETA,
    )
    if args.rms_norm_eps != EPS:
        raise SystemExit(f"TextArgs.rms_norm_eps is {args.rms_norm_eps}, expected {EPS}")
    module = KimiK3SparseMoE(args)
    rng = np.random.default_rng(0)
    drawn = [
        (name, mx.array(rng.standard_normal(value.shape).astype(np.float32) * 0.4))
        for name, value in tree_flatten(module.parameters())
    ]
    module.update(tree_unflatten(drawn))
    p = {name: np.array(value) for name, value in tree_flatten(module.parameters())}
    x = rng.standard_normal((TOKENS, 16)).astype(np.float32)
    library = np.array(module(mx.array(x)))

    scores = 1.0 / (1.0 + np.exp(-(x @ p["gate.weight"].T)))
    biased = scores + p["e_score_correction_bias"]
    chosen = np.argsort(-biased, axis=-1, kind="stable")[:, :K]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)

    gain = p["routed_expert_norm.weight"]

    def norm(v: np.ndarray, eps: float = EPS) -> np.ndarray:
        return gain * v / np.sqrt((v**2).mean(-1, keepdims=True) + eps)

    z = x @ p["routed_expert_down_proj.weight"].T
    mixed, normed_each = np.zeros_like(z), np.zeros_like(z)
    for t in range(TOKENS):
        for j in range(K):
            e = chosen[t, j]
            hidden = situ(
                z[t] @ p["switch_mlp.up_proj.weight"][e].T,
                z[t] @ p["switch_mlp.gate_proj.weight"][e].T,
            )
            out = hidden @ p["switch_mlp.down_proj.weight"][e].T
            mixed[t] += weights[t, j] * out
            normed_each[t] += weights[t, j] * norm(out)
    up = p["routed_expert_up_proj.weight"].T
    shared = (
        situ(
            x @ p["shared_experts.up_proj.weight"].T,
            x @ p["shared_experts.gate_proj.weight"].T,
        )
        @ p["shared_experts.down_proj.weight"].T
    )
    print("statement's order :", float(np.abs(library - (norm(mixed) @ up + shared)).max()))
    print("norm per expert   :", float(np.abs(library - (normed_each @ up + shared)).max()))
    print("no norm           :", float(np.abs(library - (mixed @ up + shared)).max()))
    other = norm(mixed, 1e-6) @ up + shared
    print("epsilon 1e-6      :", float(np.abs(library - other).max()))


if __name__ == "__main__":
    main()
