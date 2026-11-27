The layer carries two pieces of state because a token depends on earlier tokens in two places. Find both places in the six steps. For each one, ask what minimal tensor the next call needs so its first token produces the exact same result as a single uninterrupted call. One is a few rows of an intermediate tensor (decide whether before or after the convolution), and the other is a matrix per value head.

---

Run the convolution on `padded = concat(conv_state, P)`, where $P = x W_{qkv}^\top$. That tensor has $K - 1 + T$ rows. Output token $t$ reads rows $t, \dots, t + K - 1$, multiplying row $t + j$ by tap $j$. Set the new convolution state to `padded[T:]`, taking the last $K - 1$ rows of `padded` rather than `P` so calls with fewer than $K - 1$ tokens keep their older history.

After SiLU, slice the columns into `q`, `k`, and `v`, reshape into heads, normalize `q` and `k` with `ops.l2norm`, and scale `q` by $d_k^{-1/2}$. Expand key heads using a repeat operation (`repeat_interleave` in PyTorch, `repeat` along the head axis in JAX and MLX) so head order is `0, 0, 1, 1` rather than `0, 1, 0, 1`.

---

Run `gdn_step` in a loop over the $T$ tokens, threading `S` from the input state and returning the final `S`:

```python
for t in range(T):
    S = S * g[t, :, None, None]
    delta = beta[t, :, None] * (v[t] - einsum("hvk,hk->hv", S, k[t]))
    S = S + delta[:, :, None] * k[t, :, None, :]
    ys.append(einsum("hvk,hk->hv", S, q[t]))
```

Normalize first and gate second, per value head (so the RMS is computed over the $d_v$ channels of each head): `ops.rms_norm(y, params["norm.weight"], eps=cfg.norm_eps) * silu(z)` with `y` and `z` of shape `(T, Hv, dv)`. Finally, flatten the head axes to `(T, Hv * dv)` and apply `out_proj.weight`.
