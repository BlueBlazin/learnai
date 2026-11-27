Think of $S$ as a lookup table queried by matrix multiplication, where $S k$ is what the table currently returns for key $k$. If you want $S' k$ to move closer to $v$ by adding a rank-one matrix $u\,k^\top$, what vector $u$ achieves that? Also consider what must happen to $S$ before you ask it what it currently holds.

---

Apply the three steps in order, feeding each result into the next. First, multiply each head's matrix by its scalar $g$ by broadcasting shape `(H,)` to `(H, 1, 1)` against `(H, Dv, Dk)`. Next, compute the predicted value $\bar S k$ with a batched matrix-vector product per head, and form the correction $\beta\,(v - \bar S k)$ of shape `(H, Dv)`. Add the outer product of that correction with $k$ (shape `(H, Dv, Dk)`) to $\bar S$, and only then read $y = S' q$.

For `gdn_gates`, follow the order inside the exponent from inside out: add `dt_bias` to `a`, apply `softplus`, multiply by $e^{A_{\log}}$, negate, and exponentiate.

---

```python
S = S * g[:, None, None]
predicted = einsum("hvk,hk->hv", S, k)
delta = beta[:, None] * (v - predicted)
S = S + delta[:, :, None] * k[:, None, :]
y = einsum("hvk,hk->hv", S, q)
```

For the gates, compute `g = exp(-exp(A_log) * softplus(a + dt_bias))` and `beta = sigmoid(b)`. In MLX, `mx.logaddexp(x, 0)` is an accurate softplus. `mlx.nn.softplus` loses precision on negative inputs, though it still passes within this problem's tolerance.
