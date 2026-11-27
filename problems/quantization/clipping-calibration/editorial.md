## The idea

Symmetric quantization to $b$ bits on a range $[-t, t]$ introduces two opposing errors.

- **Rounding error** occurs inside $[-t, t]$. With $q_{\max} = 2^{b - 1} - 1$, the grid step is $t / q_{\max}$. Each in-range value is off by at most half a step, or roughly $\text{step}^2 / 12$ in mean square, which shrinks quadratically as $t$ decreases.
- **Clipping error** occurs outside $[-t, t]$. Any value $x$ with $|x| > t$ saturates at $\pm t$ and is off by $|x| - t$. This error grows as $t$ decreases, affecting only the tails of the distribution.

Min-max quantization sets $t = \max |x|$ and pays zero clipping error, which is optimal only when values fill the range uniformly. For a bell-shaped distribution with a long tail, clipping a few outliers buys a much finer grid for the bulk of the tensor. The optimal balance depends on both the data distribution and the bit width. At low bit widths, rounding error dominates and aggressive clipping helps. At 8 bits, the grid is already fine and the best range stays close to the full range.

An empirical calibration set has no closed-form optimum, and none is needed. Evaluating total MSE at each candidate range is cheap, so a direct grid search quantizes the calibration set for every candidate and keeps the one with the lowest error.

## Reference solution (PyTorch)

```python
def best_clip(samples, bits, candidates):
    qmax = 2 ** (bits - 1) - 1
    x = samples.reshape(-1)
    limit = candidates * x.abs().max()                        # (K,)
    scale = (limit / qmax)[:, None]                           # (K, 1)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)      # (K, N)
    mse = ((q * scale - x) ** 2).mean(dim=-1)                 # (K,)
    return torch.where(mse == mse.min(), candidates, torch.zeros_like(candidates)).max()
```

The JAX and MLX versions are line-for-line translations.

Three design choices make this implementation exact.

**End-to-end error measurement.** Comparing `q * scale` directly against `x` measures the exact distortion seen by the deployed model. Clamping maps out-of-range values to $\pm \text{limit}$ so clipping error enters the sum, and rounding snaps in-range values to the grid so rounding error enters the sum. Neither term needs to be modeled separately.

**One maximum across the full calibration set.** Flattening `samples` with `reshape(-1)` pools all values before taking the maximum. Because the deployed tensor uses a single scale, the calibration search must evaluate one shared range across all samples.

**Explicit tie-breaking to the larger range.** The mask `mse == mse.min()` selects every candidate that achieves the minimum error, and `.max()` picks the largest fraction among them. This result does not depend on the order of `candidates` or on how `argmin` breaks ties. Choosing the wider range on a tie is the conservative choice because it gives the same calibration error while clipping fewer unseen values at inference.

## Common mistakes

- **Minimizing clipping error only.** Comparing `clamp(x, -limit, limit)` against `x` omits rounding error. That error is zero at the full range and increases with any clipping, so the search always returns the largest candidate.
- **Forgetting the clamp.** Computing `round(x / scale) * scale` without clamping allows integer codes beyond $\pm q_{\max}$ that cannot be represented in $b$ bits. The measured error is pure rounding error on an arbitrarily fine grid, so the search always returns the smallest candidate.
- **Computing the maximum per sample.** On 2D inputs of shape `(N, D)`, calling `abs().amax(dim=-1)` gives each row its own range. That lowers calibration error artificially because quiet rows get finer grids, but the deployed tensor has only one scale.
- **Using `max` instead of `max(abs)`.** If the largest magnitude is negative, building a symmetric range from the largest positive value clips the negative tail before calibration even begins.
- **Breaking ties to the smaller range or by input order.** Using `argmin` returns the first minimum, which depends on whether `candidates` is sorted ascending or descending. Ties occur naturally at low bit widths with few distinct values, where two different ranges can yield identical squared errors.

## Complexity

Evaluating $K$ candidates over $N$ samples takes $O(K \cdot N)$ time. The vectorized implementation uses $O(K \cdot N)$ memory for the quantized table, whereas looping over candidates uses $O(N)$ memory and is preferred in production when $N$ is large.

## In the wild

**GPTQ.** The reference `Quantizer` class includes an `mse` search that scales the min-max range by `p = 1 - i / grid` for `i in range(int(maxshrink * grid))` (defaulting to `grid=100` and `maxshrink=0.8`). It starts from the full range and updates only on strictly lower error, which breaks ties in favor of the larger range. It minimizes $|\hat x - x|^{2.4}$ rather than squared error to penalize large clipping outliers more heavily.

**AWQ.** After per-channel scaling, AWQ runs `auto_clip_layer` over each weight group using `max_val * (1 - i / n_grid)` for `max_shrink * n_grid` steps (defaulting to `n_grid=20`, `max_shrink=0.5`), again scanning downward and updating on strictly lower error. Rather than measuring weight MSE, AWQ minimizes layer output MSE $\lVert x \hat w^\top - x w^\top \rVert^2$ for weights $w$, quantized weights $\hat w$, and calibration activations $x$, so weights paired with large activations carry more weight.

**Activation ranges.** Static activation quantization runs the same kind of range search on observed activations. PyTorch's `HistogramObserver` searches a histogram for the lowest quantization error, TensorRT's entropy calibrator minimizes KL divergence between original and quantized histograms, and percentile calibrators clip directly at a fixed magnitude percentile such as 99.99 percent.

**Beyond a single fraction.** A single tensor-wide clipping fraction is the coarsest form of range calibration. Group-wise quantizers calibrate a separate range per group, and learned-range methods such as LSQ and PACT optimize the clipping limit via gradient descent. All of them balance the same two errors—rounding error and clipping error—on real data.
