Think of $S$ as a lookup table that is queried by multiplication: $S k$ is "what the
table holds under $k$". If you want $S' k$ to be closer to $v$ than $S k$ was, and you
may only change $S$ by a rank-one matrix $u\,k^\top$, what does $u$ have to be? And
what must already have happened to $S$ before you ask it what it holds?

---

Do the three steps literally, each on the result of the one before. First multiply
every head's matrix by its own $g$ (shape `(H,)` against `(H, Dv, Dk)`: add two axes).
Then compute the prediction $\bar S k$ with a batched matrix-vector product, one per
head, and form the correction $\beta\,(v - \bar S k)$ of shape `(H, Dv)`. The write is
its outer product with $k$, of shape `(H, Dv, Dk)`. Only then read with $q$.

For the gates, the order inside the exponent is: add `dt_bias` to `a`, take the
softplus, multiply by $e^{A_{\log}}$, negate, exponentiate.

---

```python
S = S * g[:, None, None]
predicted = einsum("hvk,hk->hv", S, k)
delta = beta[:, None] * (v - predicted)
S = S + delta[:, :, None] * k[:, None, :]
y = einsum("hvk,hk->hv", S, q)
```

and `g = exp(-exp(A_log) * softplus(a + dt_bias))`, `beta = sigmoid(b)`. In MLX,
`mx.logaddexp(x, 0)` is an accurate softplus; `mlx.nn.softplus` loses digits for
negative inputs, which stays inside this problem's tolerance.
