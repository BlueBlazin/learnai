Start from the matrix shapes. `w` has shape `(out, in)`, `a` has shape `(r, in)`, and `b` has shape `(out, r)`. Only one product of `a` and `b` matches the shape of `w`. Identify that order and check which axis gives the rank $r$.

---

`b @ a` has shape `(out, r) @ (r, in) = (out, in)`, and $r$ is `a.shape[0]`. Multiply that product by $\alpha / r$ once. `merge` adds this scaled update to `w`, and `unmerge` subtracts the exact same term.

For `lora_forward`, apply the same transpose rule as the base layer. The base output is `x @ w.T`. Since $W' = W + \frac{\alpha}{r} B A$, the adapter contribution is `x @ (b @ a).T`, and $(B A)^{\top} = A^{\top} B^{\top}$.

---

```python
scale = alpha / a.shape[0]
merged = w + scale * (b @ a)
y = x @ w.T + scale * ((x @ a.T) @ b.T)
```

The parentheses in the last line keep the computation cheap without changing the result. Because `x @ a.T` has only `r` columns, the adapter path uses two thin matrix multiplications and never materializes an `(out, in)` matrix. Apply `scale` exactly once.
