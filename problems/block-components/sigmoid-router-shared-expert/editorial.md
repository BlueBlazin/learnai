## The idea

A softmax router makes experts compete: raising one logit lowers every other
probability. A sigmoid router scores each expert on its own, $s_e = \sigma(r_e)$, so
an expert's affinity for a token does not depend on how many other experts like it.
Competition enters only through the top-k. Four separate decisions follow, and the
code is one line each.

**Scores.** $\sigma(r)$, or $\sqrt{\operatorname{softplus}(r)}$, which is unbounded
above and close to $\sqrt{r}$ for large logits where the sigmoid saturates at 1.

**Selection uses a bias; weights do not.** The bias $b_e$ was tuned during training
to balance the load: an overloaded expert gets a lower bias, an underloaded one a
higher bias. It answers "who should take this token", which is a property of the
whole batch. The weight answers "how much does this expert know about this token",
which is the score itself. So the top-k is taken over $s + b$, and the weights are
built from $s$.

**Normalize, then scale.** Dividing the chosen scores by their sum makes the weights
a convex combination; multiplying by `scale` afterwards sets the size of the routed
contribution relative to everything else in the residual stream. In the other order
the scale is in the numerator and in the denominator and cancels:

$$
\frac{c\,s_j}{\sum_i c\,s_i} = \frac{s_j}{\sum_i s_i} .
$$

**The shared expert is outside all of it.** It sees every token, with weight 1. It is
not a $(k+1)$-th routed expert: it has no score, it is not normalized, and it is not
scaled.

The $10^{-20}$ in the denominator exists for one case. If every chosen score
underflows to 0, the weights are $0 / 10^{-20} = 0$ instead of $0/0$. It is small
enough to leave every other case alone, including scores of $10^{-13}$: an epsilon of
the usual size, `1e-6`, would not.

A note on precision: float32 `softplus` is not the same function everywhere for
very negative inputs. At a logit of $-12$ the built-in of MLX 0.32.4 and that of
PyTorch differ by 1.3 parts in $10^4$ (measured on 2026-11-21), and the textbook
`log(1 + exp(x))` is worse. That is why this problem keeps `sqrtsoftplus` logits
within $\pm 5$, where all of them agree.

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

The stable sort of the negated biased scores gives the order "largest first, ties by
lowest index" in one call. The JAX and MLX versions differ only in the names of
`gather` and `softplus`.

## Common mistakes

- **Softmax scores.** The habit from `moe-router-forward`. After renormalizing over
  the chosen experts the weights are $e^{r_j} / \sum e^{r_i}$, not
  $\sigma(r_j) / \sum \sigma(r_i)$: much sharper, and the bias, tuned for scores
  between 0 and 1, swamps probabilities that are mostly near 0.
- **Weights from the biased scores.** Gathering from `scores + bias` after the top-k
  is the shortest code and lets a load-balancing statistic rescale the experts'
  outputs. A negative bias can even make a weight negative.
- **The bias dropped,** or added to the logits instead of the scores. Both select a
  different set of experts for many tokens. $\sigma(r + b)$ is not $\sigma(r) + b$.
- **Scale before the normalization.** It cancels, and the weights sum to 1.
- **No normalization.** The raw scores times the scale: the weights of a token then
  sum to anything between 0 and $k \cdot \texttt{scale}$.
- **`softplus` without the square root.**
- **The shared expert normalized with the others, scaled, or forgotten.** With
  identical routed experts the layer must return $\texttt{scale} \cdot f(x) +
  f_{\text{shared}}(x)$, which separates all three.
- **The epsilon.** Without it, all-zero scores give NaN; with `1e-6`, tiny scores
  give tiny weights that no longer sum to the scale.
- **Ties to the highest index,** from sorting ascending and taking the last `k`.
- **An unknown score function treated as sigmoid,** in `route` or only in the layer
  that calls it. A configuration that says `sqrtsoftplus` to a loader that knows
  only `sigmoid` should stop, not route.

## Complexity

$O(N\,E\,D)$ for the router and $O(N\,E \log E)$ for the sort. The dense layer costs
$O(N\,E\,H\,D)$; an engine runs each expert on its own tokens only, $O(N\,k\,H\,D)$,
which is the subject of `moe-sparse-dispatch`.

## In the wild

Checked on 2026-11-21.

- **The paper.** "DeepSeek-V3 Technical Report" (arXiv 2412.19437), Eq. 12 to 16:
  sigmoid affinities, gating values normalized among the selected experts, shared
  experts added without gating, and a per-expert bias added to the affinities for the
  top-k selection only, with the sentence "the bias term is only used for routing".
  The scale is not in those equations.
- **The code** (`transformers` 5.23.0). `GlmMoeDsaTopkRouter` computes
  `scores = router_logits.sigmoid()`, takes the top-k of
  `scores + e_score_correction_bias`, gathers from `scores`, divides by
  `sum + 1e-20` when `norm_topk_prob` is set, and multiplies by
  `routed_scaling_factor`. `DeepseekV4TopKRouter` is the same with
  `score_fn = ACT2FN[config.scoring_func]`, and `SqrtSoftplusActivation` in
  `activations.py` is `softplus(x).sqrt()`. `GlmMoeDsaMoE` and
  `DeepseekV4SparseMoeBlock` return `routed + shared_experts(x)`.
- **The configurations.** `zai-org/GLM-5.3`: 256 routed experts, 8 per token, 1
  shared, `routed_scaling_factor` 2.5, `scoring_func` sigmoid.
  `deepseek-ai/DeepSeek-V4-Pro-0813`: 384 routed, 6 per token, 1 shared, 2.5,
  `sqrtsoftplus`. The public test "GLM-5.3 and DeepSeek-V4" compares `route` with the
  two router modules on these numbers (`fixtures/make.py`).

Where this problem is simpler than the code, or chooses for it:

- `torch.topk(..., sorted=False)` promises no order and no tie rule. The order "best
  first, ties to the lowest index" is this problem's, chosen so that the result can
  be compared exactly; the weights per expert are the library's.
- The GLM router can first restrict the choice to the best of several expert groups.
  Both configurations above have one group, where that step does nothing.
- The routed experts of DeepSeek-V4 clamp their gate and up projections
  (`swiglu_limit`), and its first layers pick experts from a table indexed by token
  id instead of by top-k. Neither is modeled here.
- `mlx-lm` 0.32.2 (`models/kimi_k3.py`, `_group_expert_select`) implements the same
  rule for Kimi's sigmoid router, with one difference: it skips the normalization
  when `top_k` is 1, so the single weight is the raw score times the scale, where
  the routers above return the scale.
