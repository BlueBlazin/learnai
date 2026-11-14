## The idea

Symmetric quantization to $b$ bits on a range $[-t, t]$ makes two kinds of error:

- **Rounding error** inside the range. The step is $t / q_{\max}$, and a rounded value
  is off by at most half a step, about $\text{step}^2 / 12$ in mean square. It shrinks
  quadratically as $t$ shrinks.
- **Clipping error** outside the range. A value $x$ with $|x| > t$ is off by
  $|x| - t$. It grows as $t$ shrinks, and it is paid only by the tail.

Min-max quantization sets $t = \max|x|$ and pays no clipping error at all, which is
optimal only if the values fill the range evenly. For a bell-shaped distribution with
a long tail, giving up a handful of outliers buys a finer grid for everything else.
The best trade depends on the distribution and on the number of bits: with few levels
rounding error dominates and aggressive clipping pays; with 8 bits the grid is already
fine and the best range is close to the full one.

There is no closed form for an empirical set of samples, and none is needed. The total
error as a function of $t$ is cheap to evaluate, so the standard method is a grid
search: quantize the calibration set at each candidate range, measure the error that
actually results, keep the best.

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

Three decisions are in these lines.

**The error is measured end to end.** `q * scale - x` is the difference between what
the deployed model will see and the original. The clamp turns out-of-range values into
$\pm\text{limit}$, so their clipping error is in the sum; the round puts in-range
values on the grid, so their rounding error is in the sum. Nothing has to be modeled
separately.

**One maximum for the whole set.** `samples.reshape(-1)` pools the calibration set
before the maximum is taken. The quantizer that ships has one scale, so the search has
to evaluate one range applied to all samples.

**Ties go to the larger range, explicitly.** `mse == mse.min()` marks every candidate
that reaches the minimum, and the `max` picks the largest of them. This does not
depend on the order of `candidates` or on which index `argmin` returns for equal
values. Preferring the larger range on a tie is the conservative choice: equal error
on the calibration set, fewer clipped values on data the calibration set did not
contain.

## Common mistakes

- **Minimizing the clipping error only.** Comparing `clamp(x, -limit, limit)` with `x`
  leaves out the rounding. That error is zero at the full range and grows with every
  bit of clipping, so the search always returns the largest candidate and is a slow
  way to do min-max.
- **Forgetting the clamp.** `round(x / scale) * scale` without the clamp never clips
  anything: outliers get codes beyond $q_{\max}$ that do not exist in $b$ bits. The
  measured error is pure rounding error on an ever finer grid, and the search always
  returns the smallest candidate. This is the same bug as the previous one, seen from
  the other side.
- **The maximum per sample.** With samples of shape `(N, D)`, taking
  `abs().amax(dim=-1)` gives every row its own range. On calibration data that looks
  better, because quiet rows get fine grids. The deployed tensor has one scale, and
  the fraction found this way is tuned for ranges that do not exist at inference.
- **`max` instead of `max(abs)`.** If the most negative value is the largest in
  magnitude, a symmetric range built from the largest positive value clips the
  negative tail before the search has started.
- **Ties to the smaller range, or to whatever comes first.** `argmin` returns the
  first minimum, which is the smallest range for an ascending list and the largest for
  a descending one. Ties look academic, but they are real at low bit widths and with
  few distinct values, where different ranges can produce the same set of errors.

## Complexity

$O(K \cdot N)$ time for $K$ candidates and $N$ samples. The vectorized version also
uses $O(K \cdot N)$ memory for the table of quantized values; a loop over candidates
needs $O(N)$ and is what calibration code usually does, since $N$ is large and $K$ is
small.

## In the wild

**GPTQ.** The reference implementation's `Quantizer` has an `mse` option: for
`i in range(int(maxshrink * grid))` it scales the min-max range by `p = 1 - i / grid`
(defaults `grid=100`, `maxshrink=0.8`), quantizes, and keeps a candidate when its
error is strictly lower than the best so far. The scan starts at the full range, so
ties stay with the larger range, as here. Its error is $|\hat x - x|^{2.4}$, not the
square: a slightly higher exponent punishes large clipping errors more.

**AWQ.** After its per-channel scaling search, AWQ runs `auto_clip_layer`: for each
group of weights it tries `max_val * (1 - i / n_grid)` for the first
`max_shrink * n_grid` steps (defaults `n_grid=20`, `max_shrink=0.5`), again scanning
from the full range down and updating on strictly lower error. The important
difference is the error it measures: not the error of the weights, but the error of
the layer's **output** on calibration activations, $\lVert x \hat w^\top - x
w^\top \rVert^2$, so weights that multiply large activations count for more.

**Activation ranges.** For static activation quantization the same search runs over
observed activations. PyTorch's `HistogramObserver` searches a histogram for the range
with the lowest quantization error, TensorRT's entropy calibrator minimizes the KL
divergence between the original and the quantized histogram, and percentile
calibrators skip the search and clip at, say, the 99.99th percentile of magnitudes.

**What a single fraction cannot do.** One clipping fraction per tensor is the coarsest
version of this idea. Group-wise formats search a range per group, and learned-range
methods (LSQ, PACT) treat the limit as a trainable parameter. The objective stays the
one in this problem: rounding error plus clipping error, measured on real data.
