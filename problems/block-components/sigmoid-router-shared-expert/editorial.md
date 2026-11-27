## The idea

A softmax router forces experts to compete before selection because raising one logit
lowers every other probability. A sigmoid router instead scores each expert
independently, $s_e = \sigma(r_e)$, so an expert's affinity for a token does not depend
on how many other experts also match that token. Competition enters only at the top-$k$
step.

**Score function.** Each logit passes through either $\sigma(r)$ or
$\sqrt{\operatorname{softplus}(r)}$. Unlike the sigmoid, which saturates at 1 for large
logits, $\sqrt{\operatorname{softplus}(r)}$ grows unboundedly like $\sqrt{r}$.

**Biased selection, unbiased weights.** The per-expert bias $b_e$ is tuned during
training to balance expert load across a batch. An overloaded expert receives a lower
bias, and an underloaded expert receives a higher bias. Selection answers which experts
should handle the token, so top-$k$ runs on $s + b$. Weighting answers how strongly a
chosen expert matches the token, so the weights use the unbiased scores $s$.

**Normalize, then scale.** Dividing the chosen scores by their sum produces a convex
combination. Multiplying by `scale` afterward sets the magnitude of the routed sum
relative to the rest of the residual stream. If you multiply by $c = \texttt{scale}$
before normalizing, $c$ cancels out of the fraction:

$$
\frac{c\,s_j}{\sum_i c\,s_i} = \frac{s_j}{\sum_i s_i} .
$$

**Independent shared expert.** The shared expert runs on every token with weight 1. It
is not treated as a $(k+1)$-th routed expert, so it has no router score, no
normalization, and no scaling.

The $10^{-20}$ term in the denominator handles one edge case. If every chosen score
underflows to 0, the weights evaluate to $0 / 10^{-20} = 0$ instead of `NaN`. At the
same time, $10^{-20}$ is small enough not to distort tiny non-zero scores around
$10^{-13}$, whereas a larger epsilon like `1e-6` would.

As a precision note, float32 `softplus` implementations diverge for very negative
inputs. At logit $-12$, MLX 0.32.4 and PyTorch differ by 1.3 parts in $10^4$ (measured
on 2026-11-21), and naive `log(1 + exp(x))` is worse. Keeping `"sqrtsoftplus"` logits
within $\pm 5$ stays in the region where all implementations agree.

## Reference solution (PyTorch)

```python
def route(x, w_router, bias, k, scale, score_fn="sigmoid"):
    logits = x @ w_router.T
    if score_fn == "sigmoid":
        scores = torch.sigmoid(logits)
    elif score_fn == "sqrtsoftplus":
        scores = torch.sqrt(F.softplus(logits))
    else:
        raise ValueError(f"unknown score_fn: {score_fn!r}")
    indices = torch.argsort(-(scores + bias), dim=-1, stable=True)[:, :k]
    chosen = torch.gather(scores, 1, indices)
    weights = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20) * scale
    return indices, weights


def moe_shared(x, w_router, bias, experts_w, shared_w, k, scale, score_fn="sigmoid"):
    w_gate, w_up, w_down = experts_w
    n_experts = w_router.shape[0]
    indices, weights = route(x, w_router, bias, k, scale, score_fn)
    one_hot = indices[..., None] == torch.arange(n_experts)
    dense = (one_hot * weights[..., None]).sum(dim=1)  # (N, E)
    y = ops.swiglu(x, *shared_w)
    for e in range(n_experts):
        y = y + dense[:, e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
```

A stable sort of the negated biased scores gives descending order with ties broken by
lowest expert index in a single call. The JAX and MLX solutions differ only in the
names of `gather` and `softplus`.

## Common mistakes

- **Using softmax scores.** Carrying over the habit from `moe-router-forward` produces
  renormalized weights $e^{r_j} / \sum e^{r_i}$ instead of
  $\sigma(r_j) / \sum \sigma(r_i)$. The distribution is much sharper, and a bias tuned
  for $[0, 1]$ sigmoid scores swamps small softmax probabilities.
- **Computing weights from biased scores.** Gathering from `scores + bias` lets a
  load-balancing statistic rescale expert outputs. A negative bias can even produce a
  negative weight.
- **Dropping the bias, or adding it to logits instead of scores.** Both change which
  experts are selected because $\sigma(r + b) \ne \sigma(r) + b$.
- **Multiplying by `scale` before normalizing.** The scale cancels in the ratio, so the
  weights sum to 1 instead of `scale`.
- **Skipping normalization.** Using `chosen * scale` directly leaves each token's
  weight sum anywhere between 0 and $k \cdot \texttt{scale}$.
- **Forgetting the square root on `softplus`.**
- **Mishandling the shared expert.** Normalizing it with routed experts, multiplying it
  by `scale`, or omitting it all fail. When every routed expert is identical, the layer
  must return $\texttt{scale} \cdot f(x) + f_{\text{shared}}(x)$, which distinguishes
  all three bugs.
- **Missing or oversized epsilon.** Omitting `1e-20` produces `NaN` when all chosen
  scores underflow to 0. Using `1e-6` shrinks tiny scores near $10^{-13}$ so the
  weights no longer sum to `scale`.
- **Breaking ties toward the highest index.** Sorting ascending and slicing the last
  `k` entries reverses the tie-breaking order.
- **Defaulting an unknown `score_fn` to `"sigmoid"`.** Either `route` or `moe_shared`
  silently falling back to sigmoid hides configuration errors instead of raising
  `ValueError`.

## Complexity

The router projection takes $O(N\,E\,D)$ time and the sort takes $O(N\,E \log E)$ time.
Dense expert evaluation takes $O(N\,E\,H\,D)$ time. A production engine runs each
expert only on its assigned tokens in $O(N\,k\,H\,D)$ time, which is covered in
`moe-sparse-dispatch`.

## In the wild

Verified on 2026-11-21.

- **Paper.** "DeepSeek-V3 Technical Report" (arXiv 2412.19437), Equations 12 to 16,
  defines sigmoid affinities, weight normalization among chosen experts, ungated shared
  experts, and a per-expert bias added only during top-$k$ selection ("the bias term is
  only used for routing"). The scaling factor does not appear in those equations.
- **Code** (`transformers` 5.23.0). `GlmMoeDsaTopkRouter` computes
  `scores = router_logits.sigmoid()`, takes top-$k$ over
  `scores + e_score_correction_bias`, gathers from `scores`, divides by `sum + 1e-20`
  when `norm_topk_prob` is enabled, and multiplies by `routed_scaling_factor`.
  `DeepseekV4TopKRouter` follows the same flow using
  `score_fn = ACT2FN[config.scoring_func]`, where `SqrtSoftplusActivation` in
  `activations.py` computes `softplus(x).sqrt()`. Both `GlmMoeDsaMoE` and
  `DeepseekV4SparseMoeBlock` return `routed + shared_experts(x)`.
- **Configurations.** `zai-org/GLM-5.3` uses 256 routed experts, 8 active per token, 1
  shared expert, `routed_scaling_factor = 2.5`, and `"sigmoid"`.
  `deepseek-ai/DeepSeek-V4-Pro-0813` uses 384 routed experts, 6 active per token, 1
  shared expert, `routed_scaling_factor = 2.5`, and `"sqrtsoftplus"`. The public test
  "GLM-5.3 and DeepSeek-V4" compares `route` against both library router modules on
  these settings (`fixtures/make.py`).

Differences between this problem and the library implementations:

- `torch.topk(..., sorted=False)` makes no guarantee on output order or tie breaking.
  This problem specifies descending order with lowest-index tie breaking so indices can
  be compared exactly, while the per-expert weights match the library.
- The GLM router supports first restricting selection to top expert groups. Both
  configurations above use a single group, so that step is a no-op.
- DeepSeek-V4 clamps its routed experts' gate and up projections (`swiglu_limit`), and
  its earliest layers select experts from a token-id lookup table instead of top-$k$.
  Neither is modeled here.
- `mlx-lm` 0.32.2 (`models/kimi_k3.py`, `_group_expert_select`) uses the same rule for
  Kimi's sigmoid router except when `top_k == 1`. In that case it skips normalization
  and returns the raw score times `scale`, whereas the routers above return `scale`.
