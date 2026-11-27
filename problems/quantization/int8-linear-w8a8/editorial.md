## The idea

Both operands are stored as integer codes that represent real numbers. For token $r$ and output row $i$, let $c_i = \text{w\_scale}_i$ and $w_{ij} = \text{w\_q}_{ij}$:

$$
x_{rj} \approx s_r\,(q_{rj} - z_r), \qquad \hat W_{ij} = c_i\, w_{ij}.
$$

Substituting these into the linear layer and factoring out every term that does not depend on the input index $j$ gives:

$$
\sum_j x_{rj} \hat W_{ij}
\approx s_r c_i \sum_j (q_{rj} - z_r)\, w_{ij}
= s_r c_i \Big( \sum_j q_{rj} w_{ij} \;-\; z_r \sum_j w_{ij} \Big).
$$

The first sum is the integer matrix product, which is the only step that scales with $\text{in}$. The second sum is a single integer per output row that can be precomputed when the weights are loaded. The float scales multiply once per output after all integer work is complete.

Three design choices and the quantization error bound follow directly from this decomposition.

**Why the weights have no zero-point.** If weights also had a zero-point $z^w_i$, expanding $(q_{rj} - z_r)(w_{ij} - z^w_i)$ would produce four terms. One of them, $z^w_i \sum_j q_{rj}$, depends on the activations and would have to be computed for every token. Keeping the weights symmetric leaves only the weight row-sum term, which is independent of `x`.

**Why the accumulator is int32.** A single product of two int8 codes can reach $128 \cdot 127 = 16256$ (15 bits), and summing 4096 such products can reach $6.7 \cdot 10^7$ (27 bits). Casting the result of an int8 matrix product to int32 is too late because the products have already wrapped.

**Why one grid per token.** The grid step is determined by the largest range it must cover. Computing one grid per token gives a token with small activations a fine step. Sharing one grid across the whole batch forces every token onto the step of the widest token and makes a token's output depend on whatever else is in the batch.

**Quantization error bound.** The weights $\hat W$ are already quantized, so they introduce no additional error. Each activation is represented within half a step, $|s_r (q_{rj} - z_r) - x_{rj}| \le s_r / 2$, including at the top clamped code. By the triangle inequality, the layer output differs from the float layer by at most $\tfrac{1}{2} s_r \sum_j |\hat W_{ij}|$.

## Reference solution (PyTorch)

```python
QMIN, QMAX = -128, 127


def int8_linear(x, w_q, w_scale, bias):
    lo = x.amin(dim=1, keepdim=True).clamp(max=0.0)
    hi = x.amax(dim=1, keepdim=True).clamp(min=0.0)
    x_scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN), torch.ones_like(hi))
    zero_point = torch.clamp(QMIN - torch.round(lo / x_scale), QMIN, QMAX).to(torch.int32)
    x_q = torch.clamp(torch.round(x / x_scale) + zero_point, QMIN, QMAX).to(torch.int8)

    w32 = w_q.to(torch.int32)
    acc = x_q.to(torch.int32) @ w32.T
    corrected = acc - zero_point * w32.sum(dim=1)
    y = corrected.to(torch.float32) * (x_scale * w_scale) + bias
    return y, acc
```

Because `lo`, `hi`, `x_scale`, and `zero_point` have shape `(n, 1)`, they broadcast along each row of `x` during quantization and against `w_scale` of shape `(out,)` during rescaling to produce `(n, out)`.

The JAX version is a direct translation. MLX differs in one line because `mx.matmul` only accepts floating-point inputs. The MLX reference computes the int32 products by broadcasting `(n, 1, in)` against `(1, out, in)` and summing over the last axis. Using a float32 `matmul` is only exact while every partial sum stays below $2^{24}$, which holds for $\text{in} \le 1032$ and fails for wider layers.

## Common mistakes

- **Accumulating in int8.** Running `x_q @ w_q.T` on int8 tensors returns int8 in PyTorch and JAX, wrapping every product modulo 256 into noise. Cast both operands to int32 before multiplying.
- **Accumulating in float32.** Float32 represents every integer exactly up to $2^{24}$, then skips values above that threshold. A layer with 2048 inputs and saturated codes sums to roughly $3.3 \cdot 10^7$, leaving the accumulator off by a few units. Because narrow layers still pass, this bug is easy to miss.
- **Sharing one activation grid across the batch.** Taking the minimum and maximum over all of `x` forces every token to use the widest token's step. This behaves identically to per-token quantization when $n = 1$, so single-token tests do not catch it.
- **Dropping the zero-point correction.** Computing `scale * acc + bias` assumes symmetric activations. With an affine grid, the output is off by $s_r c_i z_r \sum_j w_{ij}$, which is small for centered tokens but huge for non-negative tokens where $z_r = -128$.
- **Subtracting the zero-point in int8.** Computing $q - z$ before the matmul is mathematically valid, but $q - z$ spans $[-255, 255]$ and wraps in int8 whenever a token is not centered on zero.
- **Adding the bias before rescaling.** The bias is in real units, while the accumulator is in units of $s_r c_i$. Static integer engines can add a quantized bias $\operatorname{round}(b_i / (s_r c_i))$ to the accumulator offline, but a dynamic scheme cannot because $s_r$ is unknown until run time.
- **Misaligning scale axes.** `x_scale` indexes tokens and `w_scale` indexes outputs. If `x_scale` has shape `(n,)` instead of `(n, 1)`, broadcasting fails—except when `n == out`, where it silently pairs token scales with output channels.

## Complexity

Step 1 takes $O(n \cdot \text{in})$ time, step 2 performs $O(n \cdot \text{out} \cdot \text{in})$ integer multiply-accumulates, and steps 3 and 4 take $O(n \cdot \text{out})$ time. The quantized weights require 1 byte per weight plus one float32 scale per output row.

## In the wild

The `llm-compressor` presets are defined in `compressed-tensors` (`quantization/quant_scheme.py`). Its `W8A8` preset pairs static per-channel symmetric int8 weights with dynamic per-token int8 activations (`dynamic=True`), matching the per-token structure here. That preset also uses symmetric activations (`symmetric=True`), which sets every zero-point to 0 and skips step 3. This problem uses affine activations because post-nonlinearity activations are often skewed, making the zero-point correction essential whenever the grid is asymmetric.

The core integer pipeline—an int32 accumulator, a precomputed weight row-sum correction, and a final rescale—comes from *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (arXiv:1712.05877). That paper goes one step further by replacing the final float multiplication with a fixed-point multiplier and bit shift, and uses static calibration ranges so the bias can be folded into the accumulator ahead of time.
