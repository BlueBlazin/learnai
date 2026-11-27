Start from the two anchor points of the grid. There are 256 codes from $-128$ to $127$, so how many intervals lie between the smallest and largest code? If the real value `lo` must map to $-128$, what does $r = \text{scale} \cdot (q - \text{zero\_point})$ require `zero_point` to be before rounding?

---

For `quant_params`, compute the global min and max of `x`, then widen them with `min(lo, 0)` and `max(hi, 0)`. Divide the width `hi - lo` by 255, and return `scale = 1.0` when `hi == lo` to avoid dividing by zero. Solving $\text{lo} = \text{scale} \cdot (-128 - \text{zero\_point})$ gives $\text{zero\_point} = -128 - \text{lo} / \text{scale}$, which you then round and clamp to $[-128, 127]$.

For `quantize`, divide by `scale`, round, add `zero_point`, clamp to $[-128, 127]$, and cast to `int8` last. Check what your framework does if you cast an out-of-range float like `300.0` to `int8` before clamping.

---

The final trap is in `dequantize`. When `q = 127` and `zero_point = -128`, the difference is `255`, which overflows `int8`. In PyTorch, subtracting a 0-d integer tensor from an `int8` tensor stays in `int8`. Widen `q` first:

$$
r = \text{scale} \cdot \big(\operatorname{int32}(q) - \text{zero\_point}\big),
$$

then cast the difference to float32 and multiply by `scale`. Verify your code with the round trip of `0.0`. It should quantize to `zero_point`, and `(zero_point - zero_point) * scale` must equal `0.0` for any scale.
