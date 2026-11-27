Implement the activation-aware scale search from AWQ (Activation-aware Weight Quantization). Before quantizing a linear layer, AWQ scales each input channel of the weight matrix using calibration activation magnitudes and divides the activations by the same scale, preserving the unquantized function while giving high-activation weights a finer quantization grid. Your task is to search a one-parameter family of scales and return the one that minimizes the quantized layer's **output** error using only forward passes.

## Specification

```python
def awq_scales(w, x_calib, bits, group_size, grid): ...  # -> (scales, alpha)
```

Let $W$ denote the weight matrix `w` of a linear layer with shape `(out, in)`, which maps an input row $x$ to $x W^\top$. Let $X$ denote `x_calib`, a batch of $N$ calibration inputs with shape `(N, in)`.

**Group-wise quantizer.** Let $Q(\cdot)$ be the group-wise affine quantize-then-dequantize operator from `groupwise-4bit`. Each row of a weight matrix is partitioned into consecutive groups of `group_size` elements. For a group with minimum $lo$ and maximum $hi$, let $L = 2^{\text{bits}} - 1$. The step size and dequantized value are

$$
\text{step} = \begin{cases} (hi - lo) / L & hi > lo \\ 1 & hi = lo \end{cases},
\qquad
Q(v) = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{v - lo}{\text{step}}\Big),\, 0,\, L\Big) \cdot \text{step} + lo.
$$

**Scale search.** First compute the mean absolute activation of each input channel $j$:

$$
a_j = \frac{1}{N} \sum_{n=0}^{N-1} |X_{n,j}|.
$$

Then, for each exponent $\alpha$ in `grid` in order:

1. **Compute and normalize scales.** Set $s_j = \max(a_j^{\alpha},\ 10^{-4})$, using $0^0 = 1$ (the default behavior of `pow` in PyTorch, JAX, and MLX). Normalize the vector $s$ by the geometric mean of its extremes:
   $$
   s \leftarrow \frac{s}{\sqrt{\max_j s_j \cdot \min_j s_j}}.
   $$
2. **Scale, quantize, and unscale.** Compute $\hat W = Q(W \cdot s) \,/\, s$, where $s$ broadcasts along the input axis (multiplying and dividing column $j$ by $s_j$).
3. **Evaluate output error.** Compute the mean squared error between the original and quantized layer outputs on `x_calib`:
   $$
   \text{loss}(\alpha) = \frac{1}{N \cdot \text{out}} \sum_{n=0}^{N-1} \sum_{i=0}^{\text{out}-1} \Big( \big(X W^\top\big)_{n,i} - \big(X \hat W^\top\big)_{n,i} \Big)^2.
   $$

Return `(scales, alpha)` for the grid entry with the lowest loss, where `scales` is the normalized vector $s$ from step 1 and `alpha` is the matching scalar from `grid`. If multiple grid entries tie for the lowest loss, return the one that appears **first in `grid`**.

Additional input and output rules:

- `in` is divisible by `group_size`, and $N \ge 1$. A column of `x_calib` may be all zeros (a dead channel).
- `bits` is a Python int with $2 \le \text{bits} \le 8$, and `group_size` is a Python int $\ge 1$.
- `grid` contains $G \ge 1$ distinct exponents in $[0, 1]$, in **any order**.

Dividing column $j$ of the quantized weight by $s_j$ is mathematically identical to dividing activation channel $j$ by $s_j$, since $X (Q(W s) / s)^\top = (X / s)\, Q(W s)^\top$. In deployment, the inverse scale is folded into the preceding layer and only $Q(W s)$ is stored.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)` | float32 |
| `x_calib` | `(N, in)` | float32 |
| `grid` | `(G,)` | float32 |
| `scales` | `(in,)` | float32 |
| `alpha` | `()` | float32 |

## Example

```python
w = [[-0.2, 0.5, 0.7, 1.0]]
x_calib = [[1.0, 1.0, 16.0, 1.0], [-1.0, -1.0, 16.0, 1.0]]

scales, alpha = awq_scales(w, x_calib, bits=2, group_size=4, grid=[0.0, 0.5, 1.0])
# alpha = 0.5, scales = [0.5, 0.5, 2.0, 0.5]
```

The channel magnitudes are $a = [1, 1, 16, 1]$, so channel 2 is sixteen times larger than the others. The entire row forms one group, and 2-bit quantization gives $L = 3$. The exact unquantized outputs are $X W^\top = [12.5,\ 11.9]$.

| $\alpha$ | $s$ (normalized) | $W \cdot s$ | $Q(W \cdot s)$ | $\hat W = Q / s$ | $X \hat W^\top$ | loss |
|---|---|---|---|---|---|---|
| $0$ | $1,\ 1,\ 1,\ 1$ | $-0.2,\ 0.5,\ 0.7,\ 1.0$ | $-0.2,\ 0.6,\ 0.6,\ 1.0$ | $-0.2,\ 0.6,\ 0.6,\ 1.0$ | $11.0,\ 10.2$ | $2.57$ |
| $0.5$ | $0.5,\ 0.5,\ 2,\ 0.5$ | $-0.1,\ 0.25,\ 1.4,\ 0.5$ | $-0.1,\ 0.4,\ 1.4,\ 0.4$ | $-0.2,\ 0.8,\ 0.7,\ 0.8$ | $12.6,\ 11.4$ | $0.13$ |
| $1$ | $0.25,\ 0.25,\ 4,\ 0.25$ | $-0.05,\ 0.125,\ 2.8,\ 0.25$ | $-0.05,\ -0.05,\ 2.8,\ -0.05$ | $-0.2,\ -0.2,\ 0.7,\ -0.2$ | $10.6,\ 11.4$ | $1.93$ |

At $\alpha = 0$ (no scaling), weight $0.7$ rounds to $0.6$. That small $0.1$ weight error is multiplied by an activation of $16$, creating a large output error. At $\alpha = 0.5$, the raw scales $a^{0.5} = [1, 1, 4, 1]$ normalize by $\sqrt{4 \cdot 1} = 2$ to $[0.5, 0.5, 2, 0.5]$. Column 2 becomes the group maximum and is represented without error, while the coarser grid on the other three weights matters less because their activations are small. At $\alpha = 1$, column 2 stretches the group so far that the step size ($0.95$) collapses all three other weights onto the group minimum. The middle exponent $\alpha = 0.5$ achieves the lowest loss.

## Constraints

- $1 \le \text{out} \le 64$, $1 \le \text{in} \le 256$, $1 \le N \le 256$, and $G \le 20$. Entries of `w` and `x_calib` are finite with magnitude at most $10^3$.
- `alpha` must match the expected grid entry exactly. `scales` is checked with relative tolerance `1e-4` and absolute tolerance `1e-6`.
- Float32 rounding never decides a test case. In every test, either the winning loss is at least 2% below every other grid entry, or the test specifically checks ties using $\pm 1$ activations (where every $a_j = 1$, every $s_j = 1$, and all grid entries produce bit-for-bit identical losses).
- A Python loop over `grid` is expected. No loops over channels, groups, or calibration samples are needed.

## Why this matters

Standard round-to-nearest quantization minimizes weight error, but model accuracy depends on **output** error. When input channels vary widely in activation magnitude, a small rounding error on a high-activation channel causes much more damage than a larger error on a quiet channel. AWQ shows that protecting roughly 1% of channels, selected by activation magnitude rather than weight magnitude, recovers most of the accuracy lost at 3 and 4 bits without mixed-precision storage.

Because calibration takes only a few dozen forward passes per layer on a small sample set, AWQ is a standard 4-bit format in vLLM, llm-compressor, and `mlx-lm`. Common implementation bugs are easy to miss because they still return valid-looking scale vectors. Measuring error on weights instead of outputs silently picks no scaling, taking a signed activation mean produces NaNs only on negative-mean channels, and forgetting to divide back by $s$ makes every $\alpha > 0$ fail so the search falls back to $\alpha = 0$.
