## The idea

An expert's parameter count and compute scale linearly with the width $w$ that it reads and writes, because its three SwiGLU matrices have shapes $(\texttt{inter}, w)$, $(\texttt{inter}, w)$, and $(w, \texttt{inter})$. Nothing requires $w$ to equal the model width $D$. Placing one shared down-projection $D \to L$ before the experts and one shared up-projection $L \to D$ after them shrinks every expert by the factor $L / D$, while paying for the two projections only once:

$$
E \cdot 3\,D\,\texttt{inter}
\quad\longrightarrow\quad
E \cdot 3\,L\,\texttt{inter} + 2\,D\,L .
$$

What stays at full width is just as important as what moves into the latent space:

- **The router** reads the full-width token. Routing is cheap (a single $E \times D$ matrix) and benefits from seeing the complete representation.
- **The shared expert** reads and writes at full width, providing a residual path that bypasses the latent bottleneck.
- **The RMS norm** sits on the weighted mixture at latent width, right before the up-projection. The mixture is a weighted sum of $k$ expert outputs whose weights sum to 1. Normalizing it controls its scale before a single linear projection expands it back to $D$ channels.

This placement of the norm has a useful side effect. Because RMS normalization is invariant to positive rescaling of its input (up to $\epsilon$), only the *ratios* among a token's selected routing weights matter. Multiplying all of a token's weights by 2.5 barely changes $n$. A routing scale factor before this norm would be almost a no-op, which matches the shipped configuration below using a routing scale of 1.

## Reference solution (PyTorch)

```python
def latent_moe(x, w_router, w_down, w_up, norm_gain, experts, shared_w, k):
    scores = torch.sigmoid(x @ w_router.T)
    chosen = torch.argsort(-scores, dim=-1, stable=True)[:, :k]
    weights = torch.gather(scores, 1, chosen)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

    z = x @ w_down.T  # once
    mixed = torch.zeros_like(z)
    chosen_np = chosen.numpy()
    for e in range(experts.n_experts):
        token, slot = (torch.from_numpy(i) for i in np.nonzero(chosen_np == e))
        if len(token) == 0:
            continue
        out = experts(e, z[token])
        mixed = mixed.index_add(0, token, weights[token, slot][:, None] * out)

    routed = ops.rms_norm(mixed, norm_gain, eps=1e-5) @ w_up.T
    return routed + ops.swiglu(x, *shared_w)


def expert_params(hidden, latent, inter, n_experts):
    full = n_experts * 3 * hidden * inter
    with_latent = n_experts * 3 * latent * inter + 2 * hidden * latent
    return with_latent, full
```

For each expert `e`, `np.nonzero(chosen_np == e)` returns the indices of the tokens that selected it and the slot each token placed it in. Because a token selects $k$ distinct experts, each token index appears at most once per expert, so `index_add` never accumulates duplicate indices within a single call.

## Common mistakes

- **Normalizing each expert's output instead of the mixture.** Applying RMS norm inside the expert loop equalizes every expert's magnitude before summing. That computes a different function, because a weighted sum of unit-norm vectors has smaller norm than a normalized sum whenever the experts disagree.
- **Omitting the norm.** One shipped latent-MoE model includes this norm and another omits it (see below), so copying the wrong forward pass is an easy mistake.
- **Using the wrong epsilon.** This layer uses $\epsilon = 10^{-5}$ (matching the model's `rms_norm_eps`). Many other models use $10^{-6}$, and the difference becomes visible whenever the mixture has small magnitude.
- **Calling unselected experts.** Looping over all experts and calling each one even with zero rows produces the right numbers but wastes calls. This bug only surfaces when at least one expert is idle, which is the common case for short batches (such as 16 active out of 896 experts per token).
- **Dense evaluation or per-token calls.** Evaluating every expert on every token processes $E / k$ times too many rows. Calling an expert once per token makes $k \cdot N$ single-row calls instead of batching by expert.
- **Dropping routing weights or using softmax.** If weights are dropped or computed with softmax over the chosen logits, the norm hides the magnitude error, but the direction of the mixture is still wrong.
- **Forgetting the shared expert.** Dropping `ops.swiglu(x, *shared_w)` removes the only full-width path through the layer.
- **Breaking ties toward the highest index.** Use a stable descending sort so equal scores pick lower expert indices first.
- **Parameter counting errors in `expert_params`.** Counting $W_{\text{down}}$ and $W_{\text{up}}$ per expert instead of once for the layer yields 75.6 billion parameters on the Kimi K3.1 example, larger than the full-width layer it is meant to shrink. Other common bugs are forgetting the gate matrix (counting 2 matrices per expert instead of 3) or swapping the return order of the two counts.

## Complexity

Routed expert compute is $O(k\,N\,L\,\texttt{inter})$ instead of $O(k\,N\,D\,\texttt{inter})$, plus $O(N\,D\,L)$ for the two shared projections and $O(N\,E\,D)$ for the router.

## In the wild

Checked on 2026-11-21.

- **Kimi K3.1.** The text configuration in `config.json` for `moonshotai/Kimi-K3.1-Pro` specifies `hidden_size` 7168, `routed_expert_hidden_size` 3584, `moe_intermediate_size` 3072, 896 experts with 16 active per token, 2 shared experts, `latent_moe_use_norm: true`, `rms_norm_eps: 1e-5`, a sigmoid router with `moe_renormalize: true`, one expert group, and `routed_scaling_factor: 1.0`. The worked example's parameter counts are calculated from these numbers for the SwiGLU layer specified here, not measured directly from a checkpoint.
- **Reference implementation read.** `KimiK3SparseMoE` in `mlx_lm/models/kimi_k3.py` (`mlx-lm` 0.32.2) runs the router on `x`, applies `routed_expert_down_proj`, runs the experts, takes the weighted sum over chosen experts, applies `routed_expert_norm` (an RMS norm at latent width when `latent_moe_use_norm` is enabled), applies `routed_expert_up_proj`, and adds the shared expert on `x`. That file registers model type `kimi_k3` while the config above uses `kimi_k31`, and Moonshot's own modeling code was not read, so treat this correspondence as what `mlx-lm` implements rather than an official statement by the authors. `fixtures/check_mlx_lm.py` runs that module at a small size with the settings above and agrees with this problem's formulas to `1e-6`.
- **Deliberate simplifications.** In `mlx-lm`, the experts use a custom activation instead of SiLU (a `tanh`-limited gate times a `tanh`-limited up branch with `activation_situ_beta: 4.0` and `activation_situ_linear_beta: 25.0`), and the router adds a selection bias as in `sigmoid-router-shared-expert`. Its shared expert is a single MLP whose inner width is `moe_intermediate_size` times the number of shared experts. For `k = 1`, it skips weight renormalization, whereas `transformers` routers always renormalize. This problem requires $k \ge 2$, where both conventions agree.
- **Nemotron-H.** `NemotronHMoE` in `transformers` 5.23.0 (`models/nemotron_h/modeling_nemotron_h.py`) uses the same latent structure with `fc1_latent_proj` and `fc2_latent_proj` around the experts when `moe_latent_size` is set, but applies **no norm** between the mixture and the up-projection.
