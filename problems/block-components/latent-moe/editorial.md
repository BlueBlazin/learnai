## The idea

An expert's cost and size are proportional to the width it reads and writes: its
three matrices are $(\texttt{inter}, w)$, $(\texttt{inter}, w)$, and
$(w, \texttt{inter})$. Nothing forces $w$ to be the model width. Put one projection
$D \to L$ in front of all experts and one projection $L \to D$ behind them, and
every expert shrinks by the factor $L / D$, while the two projections are paid once:

$$
E \cdot 3\,D\,\texttt{inter}
\quad\longrightarrow\quad
E \cdot 3\,L\,\texttt{inter} + 2\,D\,L .
$$

What stays at full width is as important as what moves:

- **The router** reads the full-width token. Routing is cheap (one $E \times D$
  matrix), and it should see everything the token has.
- **The shared expert** reads the full-width token and writes at full width. It is
  the path that does not go through the bottleneck.
- **The norm** sits on the mixture, at latent width, before the up-projection. The
  mixture is a weighted sum of $k$ expert outputs with weights that sum to 1; the
  norm fixes its size before one linear map spreads it over $D$ channels.

The norm has a consequence worth noticing. An RMS norm is invariant to the scale of
its input (up to $\epsilon$), so only the *ratios* of a token's routing weights
survive it: multiply all of a token's weights by 2.5 and $n$ barely changes. A
routing scale in front of this norm would be nearly a no-op, which fits the shipped
configuration below having a scale of 1.

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

`np.nonzero(chosen == e)` returns, for expert `e`, the tokens that chose it and the
slot each of them chose it in; a token appears at most once, so `index_add` never
meets a repeated index within one call.

## Common mistakes

- **The norm on each expert's output.** Normalizing inside the loop gives every
  expert's contribution the same size before the weighted sum, which is a different
  function: the mixture of normalized vectors is shorter than a normalized mixture
  whenever the experts disagree.
- **The norm left out.** One shipped model with latent experts has it and one does
  not (see below), so copying the wrong forward pass is easy.
- **The epsilon.** This layer's norm uses $10^{-5}$, the model's `rms_norm_eps`.
  Many other models' norms use $10^{-6}$, and the two differ visibly as soon as the
  mixture is small.
- **Calling experts nobody chose.** A loop over all experts that calls each one, with
  zero rows if need be, returns the right output and makes wasted calls. It can only
  be told apart from the correct dispatch when some expert is idle, which with 16 of
  896 experts per token is the normal case for a short batch.
- **Dense evaluation,** or one call per token. The first does $E / k$ times the
  expert rows; the second makes $k \cdot N$ single-row calls.
- **Weights dropped,** or taken from a softmax over the chosen logits. The size of
  the mixture is repaired by the norm; its direction is not.
- **The shared expert forgotten,** which removes the only full-width path.
- **Ties** resolved to the highest index.
- **Counting the projections per expert.** They are shared: for the sizes of the
  example that mistake reports 75.6 billion parameters, more than the full-width
  layer it is supposed to shrink. Forgetting the gate matrix (two matrices per
  expert instead of three) and returning the two counts in the other order are the
  other two ways to get `expert_params` wrong.

## Complexity

Expert work is $O(k\,N\,L\,\texttt{inter})$ instead of $O(k\,N\,D\,\texttt{inter})$,
plus $O(N\,D\,L)$ for the two projections and $O(N\,E\,D)$ for the router.

## In the wild

Checked on 2026-11-21.

- **Kimi K3.1.** The `config.json` of `moonshotai/Kimi-K3.1-Pro` has, in its text
  configuration, `hidden_size` 7168, `routed_expert_hidden_size` 3584,
  `moe_intermediate_size` 3072, 896 experts with 16 per token, 2 shared experts,
  `latent_moe_use_norm` true, `rms_norm_eps` 1e-5, a sigmoid router with
  `moe_renormalize` true, one expert group, and `routed_scaling_factor` 1.0. The
  parameter counts of the worked example are computed from these numbers; they
  describe the layer of this problem (SwiGLU experts), not a measured checkpoint.
- **The code read** is `KimiK3SparseMoE` in `mlx_lm/models/kimi_k3.py` of `mlx-lm`
  0.32.2: router on `x`, `routed_expert_down_proj`, the experts, the weighted sum
  over the chosen experts, `routed_expert_norm` (an RMS norm at latent width, when
  `latent_moe_use_norm`), `routed_expert_up_proj`, and then the shared expert of
  `x` added. That file declares the model type `kimi_k3`; the configuration above
  says `kimi_k31`, and Moonshot's own modeling code was not read, so treat the
  correspondence as what `mlx-lm` implements, not as a statement by the model's
  authors. `fixtures/check_mlx_lm.py` runs that module at a small size, with the
  settings of the configuration above, against the formulas of this problem and
  agrees to 1e-6.
- **Differences from that code,** all deliberate simplifications. Its experts use an
  activation of its own in place of SiLU, a `tanh`-limited gate times a
  `tanh`-limited up branch with the configuration's `activation_situ_beta` 4.0 and
  `activation_situ_linear_beta` 25.0. Its router adds a selection bias, as in
  `sigmoid-router-shared-expert`. Its shared expert is one MLP whose inner width is
  `moe_intermediate_size` times the number of shared experts. And for `k = 1` it
  skips the normalization of the weights, where the routers of `transformers`
  normalize always; this problem requires $k \ge 2$, where the two agree.
- **Nemotron-H.** `NemotronHMoE` in `transformers` 5.23.0
  (`models/nemotron_h/modeling_nemotron_h.py`) has the same structure with
  `fc1_latent_proj` and `fc2_latent_proj` around the experts and **no norm** between
  the mixture and the up-projection, when `moe_latent_size` is set.
