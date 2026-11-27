Keep two tensors of shape `(N, E)` separate from the start: the unbiased `scores`, and
the biased `scores + bias`. Check which tensor decides *who* gets selected and which
tensor supplies the routing weights. In the worked example, expert 2 is selected with a
biased score of 0.8, so check which value enters its weight formula.

---

- **Scores.** Compute `sigmoid(x @ w_router.T)` or `sqrt(softplus(x @ w_router.T))`.
  Do not use softmax anywhere in this problem.
- **Selection.** Run a **stable** descending sort of `scores + bias` along the expert
  axis and keep the first `k` indices. Sorting the negated biased scores with a stable
  sort places larger values first and breaks ties by lowest index
  (`torch.argsort(..., stable=True)`; `jnp.argsort` and `mx.argsort` are already
  stable).
- **Weights.** Gather the **unbiased** `scores` at those indices (`gather` or
  `take_along_axis`), divide by their sum plus `1e-20`, and multiply by `scale` last.

---

```
logits  = x @ w_router.T
scores  = sigmoid(logits)                       # or sqrt(softplus(logits))
indices = argsort(-(scores + bias), stable)[:, :k]
chosen  = take_along_axis(scores, indices, -1)
weights = chosen / (chosen.sum(-1, keepdims=True) + 1e-20) * scale
```

For `moe_shared`, scatter `(indices, weights)` into a dense `(N, E)` matrix that is
zero for unselected experts, such as
`((indices[..., None] == arange(E)) * weights[..., None]).sum(1)`. Start with
`y = swiglu(x, *shared_w)`, then add
`dense[:, e:e+1] * swiglu(x, w_gate[e], w_up[e], w_down[e])` for each expert `e`. Add
the shared expert directly with no routing weight and no `scale`.
