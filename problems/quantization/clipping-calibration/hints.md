A smaller range has two opposite effects. Inside the range, grid points move closer together, so rounding error drops. Outside the range, saturated values pay their full distance to the clipping limit. Check which operation in the quantization formula produces each error, and verify that your MSE calculation includes both.

---

Try every candidate directly rather than solving analytically. First flatten `samples` and compute the single global maximum `max|samples|`. Then, for each candidate `c`, compute `limit = c * max|samples|` and `scale = limit / qmax`, quantize all samples with both `round` and `clamp`, dequantize by multiplying by `scale`, and take the mean squared difference against the original samples.

---

You can evaluate all candidates at once by adding a trailing axis to `scale`. A `scale[:, None]` tensor of shape `(K, 1)` broadcasts against flattened samples of shape `(N,)` to produce a `(K, N)` table of quantized values, and taking the mean over the last axis yields `mse` of shape `(K,)`.

For tie-breaking, do not rely on `argmin` or the order of `candidates`. Mask to the candidates whose MSE equals the minimum and take the largest:

```python
where(mse == mse.min(), candidates, 0).max()
```
