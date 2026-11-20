import mlx.core as mx
import numpy as np

from learnai.kit import CountingExperts, ops

Weights = tuple[mx.array, mx.array, mx.array]


def latent_moe(
    x: mx.array,
    w_router: mx.array,
    w_down: mx.array,
    w_up: mx.array,
    norm_gain: mx.array,
    experts: CountingExperts,
    shared_w: Weights,
    k: int,
) -> mx.array:
    # Route on the full-width input: sigmoid scores, the k best per token (the stable
    # sort ranks ties by lowest index), weights normalized over the chosen ones.
    scores = mx.sigmoid(x @ w_router.T)  # (N, E)
    chosen = mx.argsort(-scores, axis=-1)[:, :k]  # (N, k); stable
    weights = mx.take_along_axis(scores, chosen, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)

    z = x @ w_down.T  # (N, L): one projection, shared by every expert
    mixed = mx.zeros_like(z)
    chosen_np = np.array(chosen)
    for e in range(experts.n_experts):
        token, slot = (mx.array(i) for i in np.nonzero(chosen_np == e))
        if token.size == 0:
            continue  # nobody chose this expert
        out = experts(e, z[token])  # one call, its rows only, at latent width
        mixed = mixed.at[token].add(weights[token, slot][:, None] * out)

    # One norm on the mixture, still at latent width, then back up to the full width.
    routed = ops.rms_norm(mixed, norm_gain, eps=1e-5) @ w_up.T
    return routed + ops.swiglu(x, *shared_w)  # the shared expert never left full width


def expert_params(hidden: int, latent: int, inter: int, n_experts: int) -> tuple[int, int]:
    # A SwiGLU expert at width w has three matrices: gate and up (inter, w), down (w, inter).
    full = n_experts * 3 * hidden * inter
    # At latent width, plus the two projections, which exist once for all experts.
    with_latent = n_experts * 3 * latent * inter + 2 * hidden * latent
    return with_latent, full
