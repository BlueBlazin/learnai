Substitute the dequantized representations $x_{rj} \approx s_r (q_{rj} - z_r)$ and $\hat W_{ij} = \text{w\_scale}_i \cdot \text{w\_q}_{ij}$ into $\sum_j x_{rj} \hat W_{ij}$ and expand the product. Which factors do not depend on $j$ and can move outside the sum? Of the remaining terms inside the sum, which part can be computed without looking at `x`?

---

Break the layer into four steps:

1. For each row of `x`, compute the minimum and maximum with `keepdims`, widen them to include $0$, and compute the scale, zero-point, and codes as in `affine-int8`. Keep shapes as `(n, 1)` so they broadcast along each row.
2. Cast both code tensors to `int32` *before* multiplying, then compute `x_q @ w_q.T`. A product of two `int8` codes overflows `int8`, so casting after the multiply is too late.
3. Subtract `zero_point * w_q.sum(axis=1)` in `int32`.
4. Cast to `float32`, multiply by `x_scale * w_scale` (shapes `(n, 1)` and `(out,)` broadcast to `(n, out)`), and add `bias` last.

In MLX, `matmul` does not accept integer inputs. Instead, broadcast `(n, 1, in)` against `(1, out, in)` in `int32` and sum over the last axis. Do not substitute a float32 `matmul`, since float32 represents integers exactly only up to $2^{24}$ and wide layers exceed that limit.

---

$$
y_{ri} = \underbrace{s_r \cdot \text{w\_scale}_i}_{\text{float, once per output}}
\cdot \Big( \underbrace{\sum_j q_{rj}\, \text{w\_q}_{ij}}_{\text{acc, int32}}
\;-\; z_r \underbrace{\sum_j \text{w\_q}_{ij}}_{\text{int32, independent of } x} \Big)
+ \text{bias}_i .
$$

Return `acc` before subtracting the zero-point term. Because `bias` is in real units, add it after multiplying by the scales—adding it to the integer sum would wrongly scale it by $s_r \cdot \text{w\_scale}_i$.
