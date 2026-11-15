## The idea

Both operands are stored as integers that stand for real numbers:

$$
x_{rj} \approx s_r\,(q_{rj} - z_r), \qquad \hat W_{ij} = c_i\, w_{ij},
$$

with $c_i$ = `w_scale[i]` and $w_{ij}$ = `w_q[i, j]`. Substituting into the layer and
pulling everything that does not depend on $j$ out of the sum:

$$
\sum_j x_{rj} \hat W_{ij}
\approx s_r c_i \sum_j (q_{rj} - z_r)\, w_{ij}
= s_r c_i \Big( \sum_j q_{rj} w_{ij} \;-\; z_r \sum_j w_{ij} \Big).
$$

The first sum is the integer matrix product, the only work that grows with
$\text{in}$. The second is one number per output row, known when the weights are
loaded. The scales multiply once per output, after the integer work is done. That is
the whole design of integer inference: floats at the boundary, integers in the loop.

Three of the choices in the problem follow from this formula.

**Why the weights have no zero-point.** With a weight zero-point $z^w_i$ the product
$(q - z_r)(w - z^w_i)$ has four terms, and one of them, $z^w_i \sum_j q_{rj}$, depends
on the activations and must be computed for every token. With symmetric weights only
the term above remains, and it is free.

**Why the accumulator is int32.** One product of two int8 codes reaches
$128 \cdot 127 = 16256$, which needs 15 bits; a sum of 4096 of them reaches
$6.7 \cdot 10^7$, which needs 27. Casting the *result* of an int8 product to int32 is
too late: the product has already wrapped.

**Why one grid per token.** The scale is set by the largest activation the grid has to
cover. Per token, a row of small activations gets a fine grid of its own. With one grid
for the batch, that row is quantized on the step of the largest row, and its output
changes when a different sequence is batched next to it.

**The error.** The weights are exact (they *are* the codes). Each activation is
reproduced within half a step, $|s_r (q_{rj} - z_r) - x_{rj}| \le s_r / 2$, including
at the clamped top code. So the output differs from the float layer by at most
$\tfrac{1}{2} s_r \sum_j |\hat W_{ij}|$, the bound in the statement.

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

`lo`, `hi`, `x_scale`, and `zero_point` have shape `(n, 1)`, so they broadcast along a
row of `x` and, later, against `w_scale` of shape `(out,)` to give `(n, out)`.

The JAX version is a translation. MLX differs in one line: `mx.matmul` accepts
floating-point inputs only, so the reference forms the int32 products by broadcasting
`(n, 1, in)` against `(1, out, in)` and sums over the last axis. Routing the product
through a float32 `matmul` is tempting and is exact only while every partial sum stays
below $2^{24}$, which holds for $\text{in} \le 1032$ and not beyond.

## Common mistakes

- **Accumulating in int8.** `x_q @ w_q.T` on int8 tensors returns int8 in PyTorch and
  JAX: the products wrap modulo 256 and the result is noise. Cast first, multiply
  second.
- **Accumulating in float32.** Float32 represents every integer up to $2^{24}$ and
  only every second, fourth, or eighth one above. A layer with 2048 inputs and
  saturated codes sums to about $3.3 \cdot 10^7$, and the accumulator comes out a few
  units off. It is the hardest of these to notice, because every narrow layer passes.
- **One activation grid for the batch.** Taking the minimum and maximum over all of
  `x` gives every token the step of the widest one. It is identical to the per-token
  scheme for a batch of one, which is how it gets through a unit test.
- **Dropping the zero-point correction.** `scale * acc + bias` is the layer for
  activations that were quantized symmetrically. With an affine grid it is off by
  $s_r c_i z_r \sum_j w_{ij}$: small for a token centered on zero, the full output for
  a non-negative one, whose zero-point is $-128$.
- **Subtracting the zero-point in int8.** Computing $q - z$ first is a valid
  alternative to the correction term, but $q - z$ spans $[-255, 255]$ and wraps in
  int8 for every token whose range is not centered.
- **Adding the bias before the rescale.** The bias is a real number; the accumulator
  is in units of $s_r c_i$. Integer engines do add the bias to the accumulator, but
  only after quantizing it to exactly that unit, $\operatorname{round}(b_i / (s_r
  c_i))$, which a dynamic scheme cannot do offline because $s_r$ is not known.
- **Scales on the wrong axes.** `x_scale` belongs to tokens and `w_scale` to outputs.
  With `x_scale` of shape `(n,)` instead of `(n, 1)` the product fails to broadcast,
  except when `n == out`, where it silently pairs token scales with outputs.

## Complexity

Step 1 is $O(n \cdot \text{in})$ and step 2 is $O(n \cdot \text{out} \cdot
\text{in})$ integer multiply-accumulates; steps 3 and 4 are $O(n \cdot \text{out})$.
The weights take one byte each plus one float per output row.

## In the wild

The preset schemes of llm-compressor are defined in its `compressed-tensors` library
(`quantization/quant_scheme.py`). `W8A8` there is 8-bit integer weights with the
`channel` strategy, symmetric and static, and 8-bit integer input activations with
the `token` strategy and `dynamic=True`: one grid per token, computed at run time, as
in this problem. The preset sets `symmetric=True` for the activations as well, which
makes every zero-point 0 and removes step 3; this problem uses the affine form because
activations after a nonlinearity are not centered, and because the correction term is
what an integer layer must get right as soon as the grid is not symmetric.

The arithmetic itself, an int32 accumulator, a correction that is precomputed from the
weights, and a single rescale, is the scheme of *Quantization and Training of Neural
Networks for Efficient Integer-Arithmetic-Only Inference* (arXiv:1712.05877), which
also replaces the final float multiplication by a fixed-point multiplier and a shift
so that no float is needed at all. That scheme is static: the activation range comes
from calibration, so the bias can be folded into the accumulator ahead of time.
