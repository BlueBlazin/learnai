## The idea

Let $s_{i,g} = \text{scale}_{i,g}$ and $b_{i,g} = \text{bias}_{i,g}$ be the parameters for group $g$ of output row $i$. Each dequantized weight is an affine function of its code with coefficients that stay constant inside the group:

$$
\hat W_{i,j} = \text{codes}_{i,j} \cdot s_{i,g} + b_{i,g}.
$$

Substitute this into a single output $y_i$ and split the sum over inputs into a sum over groups $g$ and an inner sum over elements $j \in g$. Because $s_{i,g}$ and $b_{i,g}$ do not depend on $j$, both factor out of the inner sum:

$$
y_i = \sum_g \sum_{j \in g} x_j \big(\text{codes}_{i,j}\, s_{i,g} + b_{i,g}\big)
= \sum_g \Big( s_{i,g} \underbrace{\sum_{j \in g} x_j\, \text{codes}_{i,j}}_{\text{dots}_{i,g}}
+ \; b_{i,g} \underbrace{\sum_{j \in g} x_j}_{\text{sums}_g} \Big).
$$

This identity is exact by the distributive law and gives three advantages:

- `dots` multiplies activations by **raw codes**, which are small unsigned integers that a low-level kernel can unpack from 4-bit fields with a shift and a mask, never materializing float weights.
- Each scale is applied once per group and output row rather than once per weight.
- `sums` does not depend on the output row. With $G = \text{in} / \text{group\_size}$ groups per row, computing one sum per input group shares that work across all `out` rows, so the entire bias contribution is a single `(..., G) @ (G, out)` matrix product.

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

The JAX and MLX implementations are the same, using `astype` and `axis=`.

The reshapes establish group alignment. Reshaping `x` to `(..., G, group_size)` and `codes` to `(out, G, group_size)` splits the input axis into identical consecutive runs because row-major layout keeps adjacent elements together. `scale` and `bias` already have shape `(out, G)` in that same group order. From there, the `einsum` contracts the within-group axis `k`, multiplying by `scale` broadcasts over `(out, G)`, and summing over the last axis reduces `G`.

Casting `codes` to `float32` is an artifact of high-level array frameworks that do not multiply float and `uint8` tensors directly. That cast allocates a temporary tensor the size of `codes`, whereas a fused GPU kernel converts each code on the fly in registers as it reads it.

## Common mistakes

- **Dropping the bias.** Computing `x @ (codes * scale).T` omits the group offset. Whenever a group's minimum is far from zero, the bias accounts for most of the weight value, and the output is off by $\sum_g b_{i,g} \sum_{j \in g} x_j$.
- **Multiplying bias by `group_size`.** Because the bias appears once per element in the group, it is tempting to multiply it by the group length. However, each occurrence is weighted by its input $x_j$, so the true factor is $\sum_{j \in g} x_j$, which equals `group_size` only when every input is `1`.
- **Multiplying the sum of biases by the sum of inputs.** Computing `bias.sum(-1) * x.sum(-1)` pairs every group's bias with every group's inputs. A sum of products is not equal to a product of sums, so this works only when there is a single group per row.
- **Misaligned groups.** Reshaping to `(..., group_size, G)` and transposing, or reducing over the wrong axis, produces the expected shape `(..., G, group_size)` with strided elements (every `G`-th input) instead of consecutive runs. No shape error catches this. Test with a `group_size` that is not $\sqrt{\text{in}}$ and with one-hot inputs, which isolate single columns of $\hat W$.
- **Applying one scale per row.** Summing `dots` across groups before multiplying by `scale` treats the scale as constant along the row. That implements per-channel quantization rather than group-wise quantization.

## Complexity

Let $G = \text{in} / \text{group\_size}$ be the number of groups per row. Per row of `x`, the dot products take $O(\text{out} \cdot \text{in})$ multiply-adds, matching a standard float matmul. Scaling adds $O(\text{out} \cdot G)$ multiplications and the bias term adds $O(\text{in} + \text{out} \cdot G)$ operations, both smaller by a factor of `group_size`. In a fused kernel, weight memory traffic drops from 16 or 32 bits per weight to 4 bits per weight plus two floats per group.

## In the wild

**MLX.** `mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4)` computes `x @ W_hat.T` from codes packed into `uint32` words in the layout produced by `mx.quantize`. Every `mlx.nn.QuantizedLinear` layer calls it on every forward pass, so this identity runs once per linear layer per token. The Metal kernels unpack codes with shifts and masks during accumulation and compute `scale * accum + sum * bias` per group without ever writing the dequantized matrix to memory.

**llama.cpp.** Quantized dot-product routines consume GGUF blocks in the same way. Integer codes are multiplied and accumulated with SIMD instructions, and each block's float scale multiplies the block's partial sum. Formats that include a per-block minimum (such as `Q4_1` and the 4- and 5-bit K-quants) add the block offset multiplied by the sum of activations in that block. The paired activation format `Q8_1` precomputes and stores that block sum next to its scale specifically for this term.

**Integer-only inference.** When activations are quantized as well, `dots` becomes an integer dot product accumulated in `int32`, and both the weight and activation scales are applied afterward. The bias-times-sum term plays the exact same role as the zero-point correction term in integer-only affine matmul.
