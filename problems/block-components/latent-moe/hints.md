Follow one token through the layer and write the width of the vector after every
step: the router input, the expert input, the mixture, the normalized mixture, the
routed result, the shared expert's input and output. Only one stretch of the path is
at width $L$. Which operations lie inside it?

---

- Route from the full-width `x`: `scores = sigmoid(x @ w_router.T)`, the `k` best per
  row by a stable descending sort, weights `scores / (sum of the chosen + 1e-20)`.
- Compute `z = x @ w_down.T` **once**, before the loop over experts. Every expert
  call takes rows of `z`, never rows of `x`.
- For each expert, find the tokens that chose it (and in which of their `k` slots),
  call `experts(e, z[tokens])` once, and add `weight * out` into the rows `tokens` of
  a zero tensor of shape `(N, L)`. Skip an expert nobody chose: no call at all.

The dispatch is that of `moe-sparse-dispatch`; index bookkeeping in NumPy is fine.

---

After the loop you hold the mixture `m` of shape `(N, L)`. The norm goes here, once,
on the sum, and not on each expert's output inside the loop:

```
routed = ops.rms_norm(m, norm_gain, eps=1e-5) @ w_up.T      # (N, D)
return routed + ops.swiglu(x, *shared_w)
```

For `expert_params`, count matrices: a SwiGLU expert at width `w` has
`3 * w * inter` parameters, and the two projections have `2 * hidden * latent`
together, once for the whole layer:
`(n * 3 * latent * inter + 2 * hidden * latent, n * 3 * hidden * inter)`.
