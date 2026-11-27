Consider a single row $w$. After you round $w_0$ with error $\delta$, the layer output shifts by $\delta$ times input channel 0. Because the other input channels are correlated with channel 0, adjusting $w_1, w_2, \dots$ can cancel most of that shift. That adjustment is a least-squares problem whose closed-form solution uses rows of the upper Cholesky factor of $H^{-1}$, which the specification gives you directly. Start by building that factor cleanly, then write the column loop.

---

For `inverse_hessian_factor`, compute `h = (2 / n) * (x.T @ x)`, add `percdamp * mean(diagonal(h))` times the identity matrix, invert `h`, and take the upper-triangular Cholesky factor of the inverse (or transpose the lower factor). Inverting through a Cholesky factorization of `h` is the most stable approach (`cholesky_inverse(cholesky(h))` in PyTorch, `cholesky_inv` in MLX, or a triangular solve in JAX).

For `gptq_quantize`, compute the per-row `scale` once before the loop. Inside the loop over column `j`, read column `j` from the working weight matrix, quantize and clamp it, and store the resulting codes in a list.

---

Updating all remaining columns is a single outer product. Because row `j` of an upper-triangular `u` is zero before column `j`, subtracting the outer product from the full working matrix only modifies columns `j` and later without any slicing:

```python
q = clamp(round(w[:, j] / scale), -qmax, qmax)
err = (w[:, j] - q * scale) / u[j, j]
w = w - outer(err, u[j])
```

Column `j` itself becomes `q * scale`, which is harmless because its code is already recorded. After the loop, stack the codes along axis 1 and cast to `int8`.
