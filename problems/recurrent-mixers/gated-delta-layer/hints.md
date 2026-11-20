There are two pieces of state because there are two places where a token's result
depends on earlier tokens. Find them in the six steps. For each, ask: what is the
smallest thing the next call needs in order to compute exactly what a single long
call would have computed for its first token? One answer is a few rows of an
intermediate tensor (which one: before or after the convolution?), the other a matrix
per head.

---

Build the convolution on the concatenation of the saved rows and the new rows of
$P = x W_{qkv}^\top$: `padded = concat(conv_state, P)` has $K - 1 + T$ rows, the
output for new token $t$ reads rows $t, \dots, t + K - 1$ of it, and tap $j$ multiplies
row $t + j$. The new state is `padded[T:]`: taking the last $K - 1$ rows of `padded`
(not of `P`) is what keeps a call with one or two tokens correct.

After the SiLU, slice the columns into `q`, `k`, `v`, reshape to heads, normalize `q`
and `k` with `ops.l2norm`, scale `q`, and expand key heads with a *repeat*
(`repeat_interleave` in torch, `repeat` along the head axis in JAX and MLX), which
gives head order 0, 0, 1, 1 and not 0, 1, 0, 1.

---

The recurrence is `gdn_step` in a loop over the $T$ tokens, with `S` carried from the
state and returned after the last token:

```python
for t in range(T):
    S = S * g[t, :, None, None]
    delta = beta[t, :, None] * (v[t] - einsum("hvk,hk->hv", S, k[t]))
    S = S + delta[:, :, None] * k[t, :, None, :]
    ys.append(einsum("hvk,hk->hv", S, q[t]))
```

Then normalize first and gate second, per value head (the RMS runs over the $d_v$
channels of one head): `ops.rms_norm(y, params["norm.weight"], eps=cfg.norm_eps) *
silu(z)` with `y` and `z` of shape `(T, Hv, dv)`, flatten the two head axes, and apply
`out_proj`.
