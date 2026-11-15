## The idea

A linear layer does not care about a per-channel rescaling of its input, as long as
the weight undoes it:

$$
x W^\top = \sum_j x_j W_{\cdot j} = \sum_j \frac{x_j}{s_j}\,\big(s_j W_{\cdot j}\big)
= (x / s)\,(W s)^\top .
$$

Quantization breaks the symmetry, and that is the opening. The quantizer spends a
fixed number of levels between the minimum and the maximum of each group, so the
*absolute* rounding error of every weight in a group is about the same: up to half a
step. Scaling column $j$ up by $s_j > 1$ before quantizing and dividing by $s_j$
afterwards divides that column's absolute error by $s_j$. If the scaled column does
not become the new extreme of its group, the step is unchanged, and the column
simply got more precise for free. If it does stretch the group's range, the step
grows for everyone in the group, and the neighbors pay.

Which columns deserve the precision? The output error of one row is
$\sum_j x_j (W_{ij} - \hat W_{ij})$: each weight's error is multiplied by its
activation. So the columns to protect are the ones with large activations, and the
natural scale is a power of the channel magnitude, $s_j = a_j^\alpha$:

- $\alpha = 0$: $s = 1$, plain quantization;
- $\alpha = 1$: full scaling by the magnitude, which usually stretches the groups so
  far that the quiet columns collapse onto one or two levels (the worked example's
  last row);
- in between: the trade-off, searched on a grid, because the best exponent depends
  on the layer.

The objective is evaluated where it counts, on the layer's output for calibration
inputs, with the actual quantizer in the loop. There is nothing to differentiate:
the search is $G$ forward passes of one layer.

**The normalization.** Multiplying all of $s$ by a constant changes nothing: the
group's minimum, maximum, and step scale with it, the codes are the same, and the
constant cancels in $Q(W s) / s$. Dividing by $\sqrt{\max s \cdot \min s}$ fixes that
freedom by centering the scales around 1 in log space ($\max s \cdot \min s = 1$),
which keeps the scaled weights and the rescaled activations in a sane numeric range.
It affects the returned vector, not the loss.

**The clamp.** A channel that is always zero has $a_j = 0$ and $a_j^\alpha = 0$ for
$\alpha > 0$; the floor of $10^{-4}$ keeps the later division finite. The weights of
such a channel end up with a large error, which is multiplied by an activation of 0.

## Reference solution (PyTorch)

```python
def _quantize_groups(w, bits, group_size):
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1, keepdim=True)
    hi = groups.amax(dim=-1, keepdim=True)
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.clamp(torch.round((groups - lo) / scale), 0, levels)
    return (codes * scale + lo).reshape(out_dim, in_dim)


def awq_scales(w, x_calib, bits, group_size, grid):
    act = x_calib.abs().mean(dim=0)
    target = x_calib @ w.T
    candidates, losses = [], []
    for alpha in grid:
        s = torch.clamp(act**alpha, min=1e-4)
        s = s / torch.sqrt(s.max() * s.min())
        w_hat = _quantize_groups(w * s, bits, group_size) / s
        losses.append(((target - x_calib @ w_hat.T) ** 2).mean())
        candidates.append(s)
    best = int(torch.argmin(torch.stack(losses)))
    return candidates[best], grid[best]
```

`argmin` returns the first minimum in all three frameworks, which implements the tie
rule. The JAX and MLX versions differ only in spelling.

## Common mistakes

- **Scaling the weights only.** Comparing $X W^\top$ with $X\,Q(W s)^\top$ forgets
  the inverse scale. Every $\alpha > 0$ then looks catastrophic, because the
  compared layer computes a different function, and the search returns "no scaling".
- **Measuring the error on the weights.** $\lVert W - Q(W s)/s \rVert$ ignores the
  activations, which are the whole point; scaling a column up always hurts its
  neighbors' weight error, so this objective also drifts to $\alpha = 0$.
- **A signed mean.** `mean(x)` is near zero for the zero-mean activations inside a
  transformer and negative for half the channels; a negative number to a fractional
  power is NaN. It works on post-ReLU test data and nowhere else.
- **RMS for the mean magnitude.** $\sqrt{\operatorname{mean}(x^2)}$ is a reasonable
  statistic and a different one: it weights outliers more, changes the scales, and
  can change the winning exponent. The method specifies the mean of $|x|$.
- **Skipping or changing the normalization.** The loss is unaffected, so nothing
  looks wrong until the scales are folded into the previous layer and compared.
- **No clamp.** One dead channel gives $s_j = 0$, $W_{\cdot j} \cdot 0 / 0$ is NaN,
  the loss of every $\alpha > 0$ is NaN, and comparisons with NaN are false.
- **Quantizing before scaling.** $Q(W) \cdot s / s$ is $Q(W)$: every exponent gives
  the same loss.
- **Returning the last scales.** After the loop, `s` belongs to the last grid entry,
  not to the best one. Keep the candidates.
- **Ties to the later entry**, from `<=` in the comparison. With equal channel
  magnitudes every entry ties, and the first one is the specified answer.
- **Absolute instead of squared error.** A different objective picks a different
  exponent when a few calibration samples are much larger than the rest.

## Complexity

$G$ candidates, each one quantization, $O(\text{out} \cdot \text{in})$, and one
matrix product, $O(N \cdot \text{out} \cdot \text{in})$. With the usual grid of 20
and a few hundred calibration rows, that is a few thousand layer evaluations for a
whole model: minutes, with no backward pass and no need to hold more than one block
in memory.

## In the wild

**The method and its reference code.** "AWQ: Activation-aware Weight Quantization
for LLM Compression and Acceleration" (arXiv 2306.00978, MLSys 2024). In the
authors' repository (`mit-han-lab/llm-awq`, `awq/quantize/auto_scale.py`), the
channel statistic is `x.abs().view(-1, x.shape[-1]).mean(0)`; the grid is
`n_grid = 20` with `ratio = ratio * 1 / n_grid`, so the exponents are
$0, 0.05, \dots, 0.95$ and never reach 1; the scales are
`x_max.pow(ratio).clamp(min=1e-4)` divided by `(scales.max() * scales.min()).sqrt()`;
the weights are multiplied by the scales, pseudo-quantized, and divided by them; the
loss is the mean squared difference of the outputs; and a candidate replaces the best
only if its loss is strictly lower, so ties go to the earlier ratio. This problem
follows each of those choices. `mlx-lm`'s `mlx_lm/quant/awq.py` (version 0.32) has
the same loop in MLX.

**What is simplified here.**

- The reference searches one scale vector for a *set* of linear layers that share an
  input (for example the query, key, and value projections of an attention block),
  and measures the error on the output of the whole sub-module they feed, not on one
  matrix product.
- The scale is not kept as a separate tensor at inference. Its inverse is folded
  into whatever produces the input: the preceding normalization layer's weight, or
  the preceding linear layer. That requires nothing but a linear operation in
  between, which is why the method lists the layer pairs it scales for each
  architecture.
- Its quantizer rounds the zero point to an integer so that 0.0 is exactly
  representable. This problem uses the track's `groupwise-4bit` quantizer, which has
  a float minimum.
- After the scales, AWQ runs a second grid search over clipping ranges per group
  (`auto_clip.py`), the idea of `clipping-calibration` with the output error as the
  objective.

**Where it sits.** AWQ changes which numbers get quantized and keeps the storage
format, so its output is an ordinary group-wise 4-bit checkpoint. GPTQ, the other
calibration-based method of this track, keeps the grid and changes the rounding,
compensating each weight's error in the weights not yet quantized. The two use the
same input: a few hundred rows of activations.
