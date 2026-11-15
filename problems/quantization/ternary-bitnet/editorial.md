## The idea

Three grid points, $-s$, $0$, and $s$, split the real line at $\pm s/2$: a weight is
closer to $0$ than to $\pm s$ exactly when its magnitude is below $s / 2$. Rounding
$w / s$ to the nearest integer and clamping to $[-1, 1]$ says the same thing in one
expression, and the clamp only matters for weights beyond $1.5\,s$, which would
otherwise round to $\pm 2$ and further.

Everything then depends on $s$. BitNet b1.58 uses the mean magnitude of the matrix,
$\gamma = \operatorname{mean} |w|$. For weights that are roughly normal with standard
deviation $\sigma$, $\gamma \approx 0.8\,\sigma$, so the border sits at
$0.4\,\sigma$: about 31 percent of the weights become $0$ and about 69 percent become
$\pm 1$. The maximum magnitude, the natural choice for 8-bit grids, would be several
$\sigma$, and nearly every weight would round to $0$. With three codes there is no
room to spend a code on the outliers.

The layer follows from $\hat W = s \cdot t$:

$$
y_i = \sum_j x_j \hat W_{ij} = s \Big( \sum_{t_{ij} = 1} x_j - \sum_{t_{ij} = -1} x_j \Big).
$$

No product of an input with a weight is left, only a selection, two sums, and one
multiplication by $s$ per output. That is the claim behind "1.58-bit" inference: the
inner loop adds.

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

The JAX and MLX versions are the same lines with `jnp.where` / `mx.where` and
`expand_dims`.

The reference materializes a `(..., out, in)` tensor of selected inputs, which is fine
at the sizes of this problem and is not what a kernel does. A kernel walks the packed
codes of one row and adds or subtracts as it goes, or looks up precomputed sums of
small groups of inputs.

**The tie.** `round` in all three frameworks rounds half to even, so $\pm 0.5$ become
$0$ and the border belongs to the zero code. The paper does not say which way a tie
goes; its formula is written with a generic `round`, and the frameworks' `round` is
what an implementation of that formula does. It matters less for real weights, which
are never exactly on the border, than for the habit: `floor(r + 0.5)` and
`trunc(r + 0.5 * sign(r))` are different functions from `round`, and tests of
quantizers find out.

**The empty matrix.** The paper divides by $\gamma + \epsilon$ to survive an all-zero
matrix. This problem replaces a zero scale by 1 instead, like the other problems of
the track, so that the scale that is stored is the scale that was used.

## Common mistakes

- **Scale from the maximum.** `w.abs().max()` is the reflex from symmetric 8-bit
  quantization. Only weights above half the maximum survive, so the matrix is almost
  entirely zeros.
- **Mean over the non-zero entries.** Dividing the sum of magnitudes by the number of
  non-zero weights gives a larger scale whenever the matrix contains exact zeros
  (pruned weights, padding rows), and moves the border for every other weight. The
  mean is over all $\text{out} \cdot \text{in}$ elements.
- **The mean without the absolute value.** `w.mean().abs()` is close to 0 for any
  matrix with both signs, and every weight then saturates at $\pm 1$.
- **No clamp.** `round(w / scale)` alone produces $\pm 2$, $\pm 3$, and so on for the
  larger weights. They fit in int8, so nothing fails until the kernel, which knows
  three codes, meets a fourth.
- **Ties away from zero.** `floor(r + 0.5)` and its symmetric variant send a weight
  exactly on the border to $\pm 1$ instead of $0$.
- **Zero treated as $+1$.** Selecting the positive set with `t >= 0`, or decoding the
  codes as a sign bit, adds every input whose weight is $0$. This is the binary
  quantizer of the original BitNet, $\operatorname{sign}(w)$, applied to a ternary
  model.
- **The scale left out.** The sums alone are $x\,t^{\top}$, off by the factor
  $\gamma$. In a network with normalization layers the result can still look
  plausible, which is why it survives.

## Complexity

`ternary_quantize` is one pass for the mean and one for the codes,
$O(\text{out} \cdot \text{in})$. `ternary_matmul` performs at most
$\text{out} \cdot \text{in}$ additions per input row and one multiplication per
output. Stored at 2 bits per weight the matrix takes an eighth of float16; the
information content is $\log_2 3 \approx 1.58$ bits per weight.

## In the wild

The quantizer is equations (1) to (3) of *The Era of 1-bit LLMs: All Large Language
Models are in 1.58 Bits* (arXiv:2402.17764, 2024): the weights are divided by their
mean absolute value plus a small $\epsilon$, rounded, and clipped to $[-1, 1]$. The
same paper quantizes the activations of each layer to 8 bits, scaled per token to a
symmetric range so that no zero-point is needed; this problem leaves the activations
in float32. The paper's models are trained from scratch with the quantizer in the
forward pass. Applying it to the weights of an ordinary float model is not a
compression method: as the worked example shows, the output moves from $-4.55$ to
$-1.52$.
