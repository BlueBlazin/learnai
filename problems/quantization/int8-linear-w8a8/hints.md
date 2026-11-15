Write the real numbers that the codes stand for, $x_{rj} \approx s_r (q_{rj} - z_r)$
and $\hat W_{ij} = \text{w\_scale}_i \cdot \text{w\_q}_{ij}$, and multiply them out
inside $\sum_j x_{rj} \hat W_{ij}$. Which factors do not depend on $j$ and can leave
the sum? What is left inside it, and which part of that can be computed before any
activation is seen?

---

Four steps. (1) Per row of `x`: minimum and maximum with `keepdims`, widened to
include 0, then scale, zero-point, and codes exactly as in `affine-int8`, each of
shape `(n, 1)` so that they broadcast along the row. (2) Cast both code tensors to
int32 *before* multiplying and compute `x_q @ w_q.T`; a product of two int8 codes does
not fit in int8, and the cast has to come first. (3) Subtract
`zero_point * w_q.sum(axis=1)`, also in int32. (4) Convert to float32, multiply by
`x_scale * w_scale` (shapes `(n, 1)` and `(out,)` broadcast to `(n, out)`), and add
the bias last.

In MLX, `matmul` rejects integer inputs. Multiply `(n, 1, in)` by `(1, out, in)` in
int32 and sum over the last axis. A float32 `matmul` is not a substitute: float32
holds integers exactly only up to $2^{24}$, and a layer with thousands of inputs
exceeds that.

---

$$
y_{ri} = \underbrace{s_r \cdot \text{w\_scale}_i}_{\text{float, once per output}}
\cdot \Big( \underbrace{\sum_j q_{rj}\, \text{w\_q}_{ij}}_{\text{acc, int32}}
\;-\; z_r \underbrace{\sum_j \text{w\_q}_{ij}}_{\text{int32, independent of } x} \Big)
+ \text{bias}_i .
$$

Return `acc` as it is before the subtraction. The bias is in real units, so it is
added after the multiplication by the scales; adding it to the integer sum would
scale it by $s_r \cdot \text{w\_scale}_i$ too.
