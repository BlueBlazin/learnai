## The idea

A linear layer is invariant to per-channel rescaling of its inputs as long as the weight matrix applies the inverse scaling:

$$
x W^\top = \sum_j x_j W_{\cdot j} = \sum_j \frac{x_j}{s_j}\,\big(s_j W_{\cdot j}\big)
= (x / s)\,(W s)^\top.
$$

Quantization breaks this invariance, which creates room to reduce output error. A group-wise quantizer spreads a fixed number of levels between the minimum and maximum of each group, so every weight in the group incurs an absolute rounding error of at most half a step. Scaling column $j$ up by $s_j > 1$ before quantization and dividing by $s_j$ afterward reduces that column's effective rounding error by a factor of $s_j$. If the scaled column stays inside the group's existing range, the step size does not change and the extra precision is free. If it expands the group's range, the step size grows and neighboring columns pay the cost.

Because the output error of row $i$ is $\sum_j x_j (W_{ij} - \hat W_{ij})$, each weight's error is amplified by its input activation. Protecting channels with large activations therefore reduces output error the most. AWQ parameterizes the scale vector as a power of the mean absolute activation, $s_j = a_j^\alpha$:

- $\alpha = 0$ gives $s = 1$, which is plain quantization.
- $\alpha = 1$ scales fully by activation magnitude, which often stretches group ranges so far that low-activation weights collapse onto one or two grid points (as in the last row of the worked example).
- Values between $0$ and $1$ balance protecting loud channels against distorting quiet ones.

Because the optimal $\alpha$ varies by layer, AWQ evaluates the actual quantized output error on calibration inputs across a small grid of exponents. No gradients are needed—only $G$ forward passes of the layer.

**Normalization.** Multiplying the entire vector $s$ by any positive constant leaves quantization unchanged. Every group's minimum, maximum, and step scale by the same constant, so the integer codes stay identical and the constant cancels in $Q(W s) / s$. Dividing by $\sqrt{\max s \cdot \min s}$ removes this scale ambiguity by centering $s$ around $1$ in log space ($\max s \cdot \min s = 1$), keeping both scaled weights and rescaled activations in a well-conditioned numeric range. It changes the returned `scales` tensor without changing the loss.

**Clamping.** A dead channel that is always zero has $a_j = 0$, giving $a_j^\alpha = 0$ for $\alpha > 0$. Clamping $s_j$ from below at $10^{-4}$ prevents division by zero in $Q(W s) / s$. The weights in that column may incur a large error, which is harmless because they are multiplied by zero activations.

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

`argmin` returns the index of the first minimum in PyTorch, JAX, and MLX, which satisfies the tie-breaking rule. The JAX and MLX implementations differ only in API names.

## Common mistakes

- **Scaling the weights without unscaling.** Comparing $X W^\top$ against $X\,Q(W s)^\top$ omits the inverse scale. Every $\alpha > 0$ then computes a different linear map with huge error, so the search always picks $\alpha = 0$.
- **Measuring error on weights instead of outputs.** Minimizing $\lVert W - Q(W s)/s \rVert$ ignores activations. Because scaling any column up increases group step sizes for its neighbors, weight-space error almost always favors $\alpha = 0$.
- **Using a signed mean.** `mean(x)` is near zero for zero-mean transformer activations and negative for roughly half of the channels. Raising a negative number to a fractional power produces NaN. This bug passes on post-ReLU data and breaks everywhere else.
- **Using RMS instead of mean absolute value.** $\sqrt{\operatorname{mean}(x^2)}$ weights outliers more heavily than $\operatorname{mean}(|x|)$, producing different scales and sometimes a different winning $\alpha$.
- **Skipping or altering normalization.** Because normalization does not change the loss, the chosen $\alpha$ looks right, but the returned `scales` fail numerical checks and cannot be folded cleanly into the previous layer.
- **Omitting the $10^{-4}$ clamp.** A dead channel yields $s_j = 0$, causing $W_{\cdot j} \cdot 0 / 0$ to produce NaN. Every $\alpha > 0$ then has a NaN loss and loses all comparisons.
- **Quantizing before scaling.** Computing $Q(W) \cdot s / s$ just returns $Q(W)$, giving every $\alpha$ the exact same loss.
- **Returning the loop variable `s`.** After the loop finishes, `s` holds the scales from the last entry in `grid`, not the best one. Save each candidate or recompute `s` for the winning `alpha`.
- **Breaking ties toward the later entry.** Using `<=` instead of `<` when updating the best loss returns the last tied entry instead of the first.
- **Using absolute error instead of squared error.** Mean absolute output error weights large calibration errors differently and can pick a different exponent.

## Complexity

For $G$ grid values, each step runs one group-wise quantization in $O(\text{out} \cdot \text{in})$ time and one matrix multiplication in $O(N \cdot \text{out} \cdot \text{in})$ time. With a typical grid of 20 exponents and a few hundred calibration rows, calibrating an entire model takes a few thousand layer forward passes—a few minutes on GPU, with no backward pass and only one block in memory at a time.

## In the wild

**Reference implementation.** AWQ was introduced in "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (arXiv 2306.00978, MLSys 2024). In the authors' repository (`mit-han-lab/llm-awq`, `awq/quantize/auto_scale.py`), channel magnitudes are computed with `x.abs().view(-1, x.shape[-1]).mean(0)`. The grid uses `n_grid = 20` with `ratio = ratio * 1 / n_grid`, testing $0, 0.05, \dots, 0.95$. Scales are computed as `x_max.pow(ratio).clamp(min=1e-4)` and normalized by `(scales.max() * scales.min()).sqrt()`. Weights are scaled, pseudo-quantized, and unscaled, and the loss is the mean squared output error with strict `<` updates so earlier ratios win ties. `mlx-lm` (`mlx_lm/quant/awq.py`, version 0.32) uses the same search loop in MLX.

**Simplifications in this problem.**

- Full AWQ searches a single shared scale vector for a *group* of linear layers sharing an input (such as the Q, K, and V projections in an attention block) and evaluates output error on the entire sub-module rather than one linear layer.
- At inference time, the scale vector is not stored separately. Its inverse is folded into the preceding normalization weight or linear layer, which is why AWQ defines architecture-specific layer pairs.
- The original AWQ quantizer rounds zero points to integers so that `0.0` is represented exactly, whereas this problem uses the track's `groupwise-4bit` quantizer with a float minimum.
- After scale search, AWQ runs a second grid search for per-group weight clipping (`auto_clip.py`), applying clipping calibration directly to output error.

**How it compares.** AWQ transforms the weights before quantization while keeping the standard group-wise 4-bit storage format. GPTQ, the other calibration-based method in this track, keeps the unscaled quantization grid and adjusts the rounding of remaining weights to compensate for earlier quantization errors. Both methods require the same calibration input: a few hundred rows of activations.
