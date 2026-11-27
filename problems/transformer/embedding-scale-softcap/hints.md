There are five operations in total: a lookup, a multiplication, a matrix product, a division, and a `tanh`. The task comes down to placing each constant on the right tensor in the right order. For each constant, check which tensor it applies to and whether it multiplies or divides.

---

For `embed`, index the table with `tokens` (`table[tokens]` works across PyTorch, JAX, and MLX and preserves the shape of `tokens`), then multiply the looked-up rows by `scale`. Multiply the returned rows rather than modifying `table` in place (`table *= scale` would also scale the output head when embeddings are tied).

For `finalize_logits`, since `w_out` has shape `(V, d)`, compute the projection as `h @ w_out.T`. Divide by `logit_scale`. Then, if `softcap` is not `None`, apply the soft cap to the divided logits.

---

```python
logits = h @ w_out.T / logit_scale
if softcap is not None:
    logits = softcap * tanh(logits / softcap)
```

Dividing by `softcap` inside `tanh` and multiplying outside keeps small logits nearly unchanged ($\tanh u \approx u$) while bounding large ones near $\pm$`softcap`. Check explicitly for `None` instead of writing `if softcap:`.
