Start from the shapes. `w` is `(out, in)`, `a` is `(r, in)`, and `b` is `(out, r)`.
There is exactly one order in which `a` and `b` multiply to something of the shape of
`w`. Which one, and where does the number $r$ come from if it is not an argument?

---

`b @ a` has shape `(out, r) @ (r, in) = (out, in)`, and $r$ is `a.shape[0]`. The scale
$\alpha / r$ multiplies that product once. `merge` adds the scaled product to `w`;
`unmerge` subtracts the very same expression.

For `lora_forward`, apply the same transposition rule as for the base layer. The base
output is `x @ w.T`. Since $W' = W + \frac{\alpha}{r} BA$, the extra term is
`x @ (b @ a).T`, and $(BA)^{\top} = A^{\top} B^{\top}$.

---

```python
scale = alpha / a.shape[0]
merged = w + scale * (b @ a)
y = x @ w.T + scale * ((x @ a.T) @ b.T)
```

The parentheses in the last line matter for cost, not for the value: `x @ a.T` has
only `r` columns, so the adapter path costs two thin matmuls and the `(out, in)`
update is never built. Apply `scale` exactly once.
