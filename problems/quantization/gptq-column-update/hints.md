Start from one row $w$ and one question: after you have rounded $w_0$ and made an
error $\delta$ on it, the output is off by $\delta$ times the activations of input
0. The other inputs are correlated with input 0. How would you choose a change to
$w_1, w_2, \dots$ that cancels as much of that as possible? The answer is a least
squares problem whose solution involves $H^{-1}$, and the specification hands you its
solution in closed form. Your job is the bookkeeping: one factor up front, one loop.

---

`inverse_hessian_factor`: four lines. `h = (2 / n) * x.T @ x`; add
`percdamp * mean(diagonal(h))` times the identity; invert; take the Cholesky factor
of the inverse with `upper=True` (or transpose the lower one). Inverting through a
Cholesky factorization of `h` is the stable way: `cholesky_inverse(cholesky(h))` in
PyTorch, `cholesky_inv` in MLX, a triangular solve in JAX.

`gptq_quantize`: compute the per-row scale **once**, before the loop. Inside the
loop, read column `j` of the *working* matrix, quantize it with the clamp, and keep
the codes in a list.

---

The update of all remaining columns is one outer product. Row `j` of an
upper-triangular `u` is zero before column `j`, so subtracting the outer product
from the whole matrix touches only columns `j` and later, and no slicing is needed:

```python
q = clamp(round(w[:, j] / scale), -qmax, qmax)
err = (w[:, j] - q * scale) / u[j, j]
w = w - outer(err, u[j])
```

(Column `j` itself becomes `q * scale`, which is harmless: its code is already
stored.) After the loop, stack the codes along axis 1 and cast to int8.
