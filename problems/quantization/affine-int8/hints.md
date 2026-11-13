Start from the two anchor points. The grid has 256 codes, so how many steps lie between
the smallest code and the largest? And if the real number `lo` must land on code
$-128$, what does the equation $r = \text{scale} \cdot (q - \text{zero\_point})$ say
`zero_point` has to be, before any rounding?

---

`quant_params`: take the minimum and maximum of the whole tensor, then widen with
`min(lo, 0)` and `max(hi, 0)`. Divide the width by 255, the number of steps, and handle
the all-zero tensor (width 0) by returning scale 1 so that nothing divides by zero.
Solving $\text{lo} = \text{scale} \cdot (-128 - \text{zero\_point})$ gives
$\text{zero\_point} = -128 - \text{lo} / \text{scale}$, which must then be rounded to
an integer.

`quantize`: divide, round, add the zero-point, clamp, and only then convert to int8.
The order matters: what does your framework do when a float like `300.0` is cast to
int8?

---

The last trap is in `dequantize`. With `q = 127` and `zero_point = -128` the
difference is 255, which does not fit in int8, and integer arithmetic on an int8 tensor
stays in int8 (in PyTorch even when the other operand is a 0-d int64 tensor). Convert
`q` to a wider type first:

$$
r = \text{scale} \cdot \big(\operatorname{int32}(q) - \text{zero\_point}\big),
$$

then convert to float32 and multiply. Check yourself with the round trip of `0.0`:
it quantizes to `zero_point`, and `(zero_point - zero_point) * scale` is exactly `0.0`
for every scale.
