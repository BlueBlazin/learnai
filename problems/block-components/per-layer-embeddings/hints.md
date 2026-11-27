Start with a small helper for the norm because every RMS norm in this problem uses `ops.rms_norm(x, gain, eps=1e-6)`. Next, build `ple_inputs` as two `(T, L * d_ple)` tensors before splitting by layer: the scaled table rows and the scaled projection of `h0`. What single `reshape` makes `[:, i]` equal to columns `i * d_ple` through `(i + 1) * d_ple - 1`?

---

`reshape(T, L, d_ple)` splits each row into consecutive slices of length `d_ple`, matching the packed layout. By contrast, `reshape(T, d_ple, L)` deals columns across layers like cards.

For the projected part, order matters when inputs are small. Multiply `h0 @ w_model.T` by `D ** -0.5` first, reshape to `(T, L, d_ple)`, and then normalize each slice over its `d_ple` channels with `norm_gain`. Normalize only the projected part. Scale the table part by `sqrt(d_ple)` (not `sqrt(D)`), add the two parts, and multiply the sum by `2 ** -0.5`.

---

Once tanh-GELU is implemented from the formula, `ple_inject` takes two lines:

```python
gated = gelu(h @ w_gate.T) * ple_i  # (T, d_ple)
return h + ops.rms_norm(gated @ w_proj.T, post_gain, eps=1e-6)
```

Avoid `h += ...` so you do not mutate the caller's tensor in place.

---

`ple_layer` applies the three equations in sequence. Both attention and the MLP follow the pattern norm, sublayer, norm, residual add, using four separate gain vectors. Pass `m` (the stream after the MLP's residual add) and `ple[:, i]` into `ple_inject`. Multiply by `params["layer_scalar"]` at the very end so it scales both the residual stream and the injected branch.
