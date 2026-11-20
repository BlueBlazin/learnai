Unroll the step from a zero state for three tokens and collect, for the output at
$t$, the coefficient of each earlier input $x_s$. Three things multiply it: how the
input was written (which vector, which scale), how often the state was decayed
between $s$ and $t$ (was it decayed by token $s$'s own step?), and how it is read.

---

For the step: compute `dt` with softplus and then the clamp (a clamp on the raw value
is a different function), expand `B` and `C` from groups to heads with a *repeat*
along the group axis (`repeat_interleave` in torch, `repeat` in JAX and MLX), and
update `h * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]`.
Read the *new* `h` with `C`.

For the matrix: the decay between $s$ and $t$ is a difference of running sums. With
$c_t = -e^{A_{\log}} \sum_{r \le t} \Delta_r$, the factor is $e^{c_t - c_s}$, which
excludes $\Delta_s$ and includes $\Delta_t$ as required.

---

```python
scores = einsum("thn,shn->hts", C, B)                 # after repeating groups to heads
c = cumsum(-dt * exp(A_log), axis=0).T                # (H, T)
decay = exp(where(t >= s, c[:, :, None] - c[:, None, :], -inf))
M = scores * decay * dt.T[:, None, :]                 # dt of the source token s
```

Masking the exponent with `-inf` gives exact zeros above the diagonal. In MLX,
`mx.logaddexp(u, 0)` is an accurate softplus; `mlx.nn.softplus` is accurate enough
here because small step sizes are either clamped or compared loosely.
