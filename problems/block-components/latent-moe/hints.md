Trace a single token through the layer and write down the vector width after each step: router input, expert input, weighted mixture, normalized mixture, up-projected routed output, and shared expert input and output. Only one segment of the path operates at width $L$. Which operations belong inside that segment?

---

- Route from the full-width `x`. Compute `scores = sigmoid(x @ w_router.T)`, pick the `k` largest entries per row using a stable descending sort, and normalize them with `scores / (sum of chosen + 1e-20)`.
- Compute `z = x @ w_down.T` **once**, before looping over experts. Pass rows of `z` to `experts`, never rows of `x`.
- For each expert `e`, find the tokens that chose it and the slot each token used. If no token chose `e`, skip it without calling `experts`. Otherwise, call `experts(e, z[tokens])` once and add `weight * out` into rows `tokens` of a zero tensor of shape `(N, L)`.

This is the same sparse dispatch pattern as `moe-sparse-dispatch`, and doing the index bookkeeping in NumPy is fine.

---

After the loop finishes, you have the mixture `m` of shape `(N, L)`. Apply the RMS norm once to `m` here, not to individual expert outputs inside the loop:

```python
routed = ops.rms_norm(m, norm_gain, eps=1e-5) @ w_up.T  # (N, D)
return routed + ops.swiglu(x, *shared_w)
```

For `expert_params`, count the matrices directly. Each SwiGLU expert at width `w` has `3 * w * inter` parameters, and the two shared projections add `2 * hidden * latent` once for the whole layer:
`(n_experts * 3 * latent * inter + 2 * hidden * latent, n_experts * 3 * hidden * inter)`.
