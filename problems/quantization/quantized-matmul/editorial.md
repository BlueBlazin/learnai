## The idea

The dequantized weight is affine in the code, with coefficients that are constant
inside a group:

$$
\hat W_{i,j} = \text{codes}_{i,j} \cdot s_{i,g} + b_{i,g} .
$$

Substitute that into one output and split the sum over inputs into a sum over groups
and a sum inside each group. The scale and the bias leave the inner sum:

$$
y_i = \sum_g \sum_{j \in g} x_j \big(\text{codes}_{i,j}\, s_{i,g} + b_{i,g}\big)
= \sum_g \Big( s_{i,g} \underbrace{\sum_{j \in g} x_j\, \text{codes}_{i,j}}_{\text{dots}_{i,g}}
+ \; b_{i,g} \underbrace{\sum_{j \in g} x_j}_{\text{sums}_g} \Big).
$$

Nothing is approximated; this is the distributive law. What it buys:

- `dots` multiplies activations by **raw codes**, small unsigned integers that a kernel
  can unpack from 4-bit fields with a shift and a mask. No float weight ever exists.
- The scale is applied once per group and output, not once per weight.
- `sums` does not depend on the output row. One sum per group of inputs is shared by
  all `out` rows, so the whole bias term costs a `(G) x (G, out)` product.

## Reference solution (PyTorch)

```python
def qmatmul(x, codes, scale, bias, group_size):
    out_dim, in_dim = codes.shape
    n_groups = in_dim // group_size
    xg = x.reshape(*x.shape[:-1], n_groups, group_size)            # (..., G, g)
    cg = codes.reshape(out_dim, n_groups, group_size).to(x.dtype)  # (O, G, g)
    dots = torch.einsum("...gk,ogk->...og", xg, cg)                # (..., O, G)
    sums = xg.sum(dim=-1)                                          # (..., G)
    return (dots * scale).sum(dim=-1) + sums @ bias.T
```

The JAX and MLX versions are the same with `astype` and `axis=`.

The reshapes are where alignment is decided. `x.reshape(..., G, group_size)` and
`codes.reshape(out, G, group_size)` cut the input axis the same way, into consecutive
runs, because a row-major reshape keeps neighbors together. `scale` and `bias` already
have a `G` axis in the same order. After that, every operation matches axes by name:
the einsum contracts the within-group axis `k`, the multiplication by `scale` lines up
`(out, G)`, and the final sum removes `G`.

The cast of the codes to float32 is a concession to array frameworks, which cannot
multiply a float tensor by a uint8 one. It is a temporary the size of the codes, which
a real kernel does not need: it converts each code as it reads it.

## Common mistakes

- **Dropping the bias.** `x @ (codes * scale).T` looks plausible because the codes are
  "the weights". The bias is most of the weight's value whenever a group does not
  start near zero, and the result is off by $\sum_g b_{i,g} \sum_{j \in g} x_j$.
- **Bias times `group_size`.** The bias appears once per element of the group, so
  multiplying it by the element count feels right. But each appearance is weighted by
  its input: the factor is $\sum_{j \in g} x_j$, which equals `group_size` only for an
  input of all ones.
- **Sum of biases times sum of inputs.** `bias.sum(-1) * x.sum(-1)` pairs every bias
  with every group's inputs. A sum of products is not a product of sums; this is right
  only with a single group per row.
- **Misaligned groups.** `x.reshape(..., group_size, G)` followed by a transpose, or a
  reduction over the wrong axis, yields the right shape, `(..., G, group_size)`, with
  the wrong content: every `G`-th input instead of `group_size` consecutive ones. No
  shape error will ever tell you. Test with a group size that is not the square root
  of `in`, and with one-hot inputs, which read out single columns of $\hat W$.
- **One scale per row.** Summing the dot products over the groups before scaling
  treats the scale as constant along a row. That is per-channel quantization; the
  whole point of groups is that the scale changes every `group_size` inputs.

## Complexity

The dot products cost $O(\text{out} \cdot \text{in})$ multiply-adds per row of `x`,
the same as a float matmul. The scales add $O(\text{out} \cdot G)$ multiplications and
the bias term $O(\text{in} + \text{out} \cdot G)$, both smaller by a factor of
`group_size`. In a kernel, the memory traffic for the weights drops from 32 (or 16)
bits to 4 bits per weight plus two floats per group, which is the reason to do it.

## In the wild

**MLX.** `mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64,
bits=4)` computes `x @ W_hat.T` from `w` packed in uint32 words, for the layout that
`mx.quantize` produces. `mlx.nn.QuantizedLinear` calls it in every forward pass, so
this identity runs once per linear layer per token in every quantized MLX model. Its
kernels unpack the codes with shifts and masks as they accumulate, and return
`scale * accum + sum * bias` per group, never writing the dequantized matrix.

**llama.cpp.** GGUF blocks are consumed the same way by the quantized dot-product
routines: integer codes are multiplied and accumulated with SIMD, and the block's
float scale is applied to the block's partial sum. Formats with a per-block minimum
(`Q4_1`, the 4- and 5-bit K-quants) need the equivalent of the bias term, the block's
offset times the sum of the activations over that block. The activation format they
are paired with, `Q8_1`, stores that sum next to its scale for exactly this purpose.

**Integer-only inference.** When the activations are quantized too, `dots` becomes an
integer dot product accumulated in int32 and both scales are applied afterward. The
bias-times-sum term is the same correction that appears there as the zero-point term.
