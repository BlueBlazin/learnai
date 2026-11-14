A smaller range has two opposite effects. For values inside the range, the grid points
move closer together, so rounding error goes down. For values outside it, the error is
the whole distance to the limit. Which line of the quantization formula produces each
of these two errors, and does your error measure include both?

---

There is nothing to solve analytically: try every candidate. For one candidate,
compute `limit = c * max|samples|` and `scale = limit / qmax`, quantize all samples
with round **and** clamp, dequantize by multiplying with `scale` again, and take the
mean of the squared differences to the original samples. The maximum is taken once,
over the whole flattened calibration set, before the loop.

---

You can evaluate all candidates at once by giving the scales a trailing axis:
`scale[:, None]` has shape `(K, 1)` and broadcasts against the flattened samples
`(N,)` to a `(K, N)` table of quantized values, and a mean over the last axis gives
`mse` of shape `(K,)`.

For the tie rule, do not rely on what `argmin` does with equal values or on the order
of the list. Select the candidates whose error equals the minimum, and take the
largest:

```python
where(mse == mse.min(), candidates, 0).max()
```
