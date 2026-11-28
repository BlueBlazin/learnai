## The idea

Let $s$ be the scale of the ternary grid. The three representable values $-s$, $0$, and $s$ divide the real line at $\pm s / 2$. A weight is closer to $0$ than to $\pm s$ whenever its magnitude is less than $s / 2$. Rounding $w / s$ to the nearest integer and clamping to $[-1, 1]$ expresses this rule in one line, where the clamp handles weights beyond $1.5\,s$ that would otherwise round to $\pm 2$ or further.

The choice of $s$ determines how weights split across the three codes. BitNet b1.58 sets $s$ to the mean absolute value of the matrix, $\gamma = \operatorname{mean}|w|$. For weights that are roughly normal with standard deviation $\sigma$, we have $\gamma \approx 0.8\,\sigma$, which places the threshold at $0.4\,\sigma$. About 31% of the weights map to $0$ and about 69% map to $\pm 1$. By contrast, using the maximum magnitude, as in symmetric 8-bit quantization, would set $s$ to several $\sigma$, causing nearly every weight to round to $0$. With only three codes, you cannot afford to scale the grid around outliers.

The linear layer follows directly from $\hat W = s \cdot t$:

$$
y_i = \sum_j x_j \hat W_{ij} = s \Big( \sum_{t_{ij} = 1} x_j - \sum_{t_{ij} = -1} x_j \Big).
$$

No input-weight multiplication remains. Each output requires only selecting inputs, computing two sums, and multiplying once by $s$. That is the hardware motivation behind 1.58-bit inference.

## Reference solution (PyTorch)

```python
def ternary_quantize(w):
    gamma = w.abs().mean()
    scale = torch.where(gamma > 0, gamma, torch.ones_like(gamma))
    t = torch.clamp(torch.round(w / scale), -1, 1)
    return t.to(torch.int8), scale


def ternary_matmul(x, t, scale):
    xe = x.unsqueeze(-2)  # (..., 1, in) against t of shape (out, in)
    zero = torch.zeros_like(xe)
    plus = torch.where(t == 1, xe, zero).sum(dim=-1)
    minus = torch.where(t == -1, xe, zero).sum(dim=-1)
    return scale * (plus - minus)
```

The JAX and MLX implementations use the same operations with `jnp.where` or `mx.where` and `expand_dims`.

This reference materializes a `(..., out, in)` tensor of selected inputs, which is fine for the problem's constraints. A production kernel instead streams through packed codes row by row, accumulating additions and subtractions in registers or looking up precomputed sums of small input groups.

**Ties on the border.** In all three frameworks, `round` uses round-to-nearest-even, so $\pm 0.5$ map to $0$. The BitNet paper writes a generic `round` without specifying tie-breaking, so using the framework's `round` is the standard choice. While real trained weights rarely land exactly on $\pm 0.5$, custom helpers like `floor(r + 0.5)` or `trunc(r + 0.5 * sign(r))` behave differently on ties and fail border tests.

**All-zero matrices.** The BitNet paper divides by $\gamma + \epsilon$ to avoid dividing by zero. Here we replace a zero scale with $1$ instead, matching the rest of the track so that the returned `scale` is the exact value used in quantization.

## Common mistakes

- **Scale from the maximum.** Using `w.abs().max()` is a habit from symmetric 8-bit quantization. Only weights larger than half the maximum survive, so almost the entire matrix becomes zero.
- **Mean over non-zero entries only.** Dividing the sum of magnitudes by the count of non-zero weights inflates `scale` whenever `w` has exact zeros (such as pruned weights or padding rows), shifting the border for every other weight. The mean must cover all $\text{out} \cdot \text{in}$ elements.
- **Mean without taking the absolute value.** Computing `w.mean().abs()` produces a number near $0$ whenever positive and negative weights cancel out, saturating almost every weight to $\pm 1$.
- **Missing clamp.** Computing `round(w / scale)` without clamping produces values like $\pm 2$ or $\pm 3$. Because they still fit in `int8`, the bug hides until a kernel expecting only three codes encounters them.
- **Ties away from zero.** Using `floor(r + 0.5)` or its symmetric variant sends border weights ($\pm 0.5$) to $\pm 1$ instead of $0$.
- **Treating zero as $+1$.** Selecting positive entries with `t >= 0` or decoding weights as a single sign bit adds inputs whose weight is $0$. That applies binary BitNet ($\operatorname{sign}(w)$) to a ternary model.
- **Forgetting to multiply by `scale`.** Returning `plus - minus` computes $x t^{\top}$, which is off by a factor of $\gamma$. In networks with normalization layers, a missing global scale can look deceptively plausible.

## Complexity

`ternary_quantize` makes one pass to compute the mean and one pass to compute the codes, taking $O(\text{out} \cdot \text{in})$ time. `ternary_matmul` performs at most $\text{out} \cdot \text{in}$ additions per input row and one multiplication per output. Packed at 2 bits per weight, the matrix uses one-eighth the memory of float16, while its information content is $\log_2 3 \approx 1.58$ bits per weight.

## In the wild

This quantizer matches equations (1) through (3) of *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (arXiv:2402.17764, 2024), where weights are divided by their mean absolute value plus a small $\epsilon$, rounded, and clipped to $[-1, 1]$. The paper also quantizes activations to 8 bits using a symmetric per-token scale, whereas this problem keeps activations in float32. Because BitNet b1.58 models are trained from scratch with this quantizer in the forward pass, applying it post-hoc to a standard float model is not a viable compression method. In the worked example, doing so shifts the first output from $-4.55$ to $-1.52$.
