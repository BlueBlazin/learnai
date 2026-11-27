The quantization math is standard affine quantization applied to short runs of weights instead of an entire tensor. Think about tensor shapes first. How can you reshape an `(out, in)` matrix so that every run of `group_size` consecutive elements in a row sits on its own axis? That lets you compute group minima and maxima without a loop.

---

Reshape `w` to `(out, in // group_size, group_size)`. Row-major order keeps consecutive elements of each row adjacent, so the last axis is one group. Taking the `min` and `max` over the last axis gives `bias` and the upper bound in the exact shape `(out, in // group_size)`. Add a trailing axis of size 1 (`[..., None]`) to broadcast `scale` and `bias` back against the reshaped groups, and remember that $2^{\text{bits}}$ codes span $2^{\text{bits}} - 1$ steps.

---

Let `groups` be the reshaped weight matrix, and let `lo` and `hi` be its group minima and maxima. Guard against constant groups with `where` before dividing. Use `(hi - lo) / (2**bits - 1)` where `hi > lo`, and `1` elsewhere so no division by zero occurs. Then compute:

$$
\text{codes} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{\text{groups} - \text{lo}}{\text{scale}}\Big),\, 0,\, 2^{\text{bits}} - 1\Big).
$$

Cast `codes` to `uint8` after clamping, and reshape back to `(out, in)`.
