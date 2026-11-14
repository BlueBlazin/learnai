The three formulas are the affine quantization you already know, applied to a short run
of weights instead of a whole tensor. So the real question is one of shapes: how do you
get the minimum and the maximum of every run of `group_size` consecutive elements of
every row, without a loop? Which reshape of an `(out, in)` matrix puts each group on an
axis of its own?

---

Reshape `w` to `(out, in // group_size, group_size)`. Row-major order keeps consecutive
elements of a row together, so the last axis now is one group, and a `min` and a `max`
over the last axis give `bias` and the range with exactly the shapes you must return.
To quantize, broadcast them back against the groups by adding a trailing axis of
length 1. Between $2^{\text{bits}}$ codes there are $2^{\text{bits}} - 1$ steps.

---

The constant group is the only branch. Where `hi > lo`, the scale is
`(hi - lo) / (2**bits - 1)`; elsewhere it is 1, chosen with `where` so that no division
by zero is ever evaluated on the path you keep:

$$
\text{codes} = \operatorname{clamp}\Big(\operatorname{round}\Big(\frac{\text{groups} - \text{lo}}{\text{scale}}\Big),\, 0,\, 2^{\text{bits}} - 1\Big).
$$

Cast to uint8 after the clamp, and reshape the codes back to `(out, in)`.
