Unroll `ssd_step` from a zero state for three tokens and collect the coefficient of each earlier input $x_s$ in the output $y_t$. Three factors multiply $x_s$: how the input was written (which vector and scale), how many decay steps occurred between $s$ and $t$ (check whether token $s$'s own step decays its own write), and how the state is read at $t$.

---

For `ssd_step`, compute `dt` by applying softplus first and clamping second, since clamping the raw value defines a different function. Expand `B` and `C` from groups to heads by repeating along the group axis (`repeat_interleave` in PyTorch, `repeat` in JAX and MLX). Update the state with `h * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]`, then read the updated `h` with `C`.

For `ssd_matrix`, express the decay between $s$ and $t$ as a difference of cumulative sums. With $c_t = -e^{A_{\log}} \sum_{r \le t} \Delta_r$, the factor $e^{c_t - c_s}$ automatically excludes $\Delta_s$ and includes $\Delta_t$.

---

```python
scores = einsum("thn,shn->hts", C, B)                 # after repeating groups to heads
c = cumsum(-dt * exp(A_log), axis=0).T                # (H, T)
decay = exp(where(t >= s, c[:, :, None] - c[:, None, :], -inf))
M = scores * decay * dt.T[:, None, :]                 # dt of the source token s
```

Masking the exponent with `-inf` produces exact zeros above the diagonal. In MLX, `mx.logaddexp(u, 0)` gives an accurate softplus, though `mlx.nn.softplus` is also accurate enough here because very small step sizes are either clamped or checked with looser tolerances.
