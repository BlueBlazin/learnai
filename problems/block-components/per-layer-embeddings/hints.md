Write the three norms' worth of helper first: every norm in this problem is
`ops.rms_norm(x, gain, eps=1e-6)`. Then do `ple_inputs` in two passes of the same
shape, `(T, L * d_ple)`, before you think about layers: the scaled table rows and the
scaled projection of `h0`. What does one `reshape` have to do so that `[:, i]` is
columns `i * d_ple` to `(i + 1) * d_ple - 1`?

---

`reshape(T, L, d_ple)` cuts a row into consecutive slices, which is the layout of the
packed row. `reshape(T, d_ple, L)` would deal the columns out like cards.

The order of operations on the projected part matters once the numbers are small:
multiply by `D ** -0.5` first, reshape, then normalize each slice over its `d_ple`
channels with `norm_gain`. Only the projected part is normalized. The table part is
only scaled, by `sqrt(d_ple)` (not by `sqrt(D)`, the scale of the main embedding). The
sum is multiplied by `2 ** -0.5`.

---

`ple_inject` is one line once GELU is written from the formula:

```python
gated = gelu(h @ w_gate.T) * ple_i  # (T, d_ple)
return h + ops.rms_norm(gated @ w_proj.T, post_gain, eps=1e-6)
```

Do not use `+=` on `h`: the caller's tensor must stay as it was.

---

`ple_layer` is the three equations in order. Both sublayers are "norm, sublayer, norm,
add"; the four gains are four different vectors. The injection reads `m`, the stream
after the MLP's add, and takes `ple[:, i]`. Multiply by `params["layer_scalar"]` last,
so that it scales the stream and the injected branch alike.
