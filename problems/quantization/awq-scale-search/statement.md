Implement the scale search of AWQ (activation-aware weight quantization): before a
linear layer is quantized, each input channel of its weight is multiplied by a scale
derived from how large that channel's activations are on calibration data, and the
activations are divided by the same scale, so the layer computes the same function
but the important weights sit on a finer grid. You search a one-parameter family of
scales for the one whose quantized layer has the lowest **output** error. This is
post-training calibration: no gradients, no optimizer, only forward passes.

## Specification

```python
def awq_scales(w, x_calib, bits, group_size, grid): ...  # -> (scales, alpha)
```

`w` is the weight of a linear layer, shape `(out, in)`, which maps an input row
$x$ to $x W^\top$. `x_calib` holds $N$ calibration inputs of that layer, shape
`(N, in)`.

**The quantizer.** $Q(\cdot)$ is the group-wise affine quantizer of
`groupwise-4bit`, followed by dequantization. Each row is cut into consecutive
groups of `group_size` inputs. For a group with minimum $lo$ and maximum $hi$, and
$L = 2^{\text{bits}} - 1$:

$$
\text{step} = \begin{cases} (hi - lo) / L & hi > lo \\ 1 & hi = lo \end{cases},
\qquad
Q(v) = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{v - lo}{\text{step}}\Big),\, 0,\, L\Big) \cdot \text{step} + lo .
$$

**The search.** For each exponent $\alpha$ in `grid`, in order:

1. **Channel magnitudes.** $a_j = \frac{1}{N} \sum_{n} |x_{n,j}|$, the mean absolute
   activation of input channel $j$.
2. **Scales.** $s_j = \max(a_j^{\alpha},\ 10^{-4})$, then normalize:
   $$
   s \leftarrow \frac{s}{\sqrt{\max_j s_j \cdot \min_j s_j}} .
   $$
   Use $0^0 = 1$, which is what `pow` returns in every framework.
3. **Scale, quantize, unscale.** $\hat W = Q(W \cdot s) \,/\, s$, where $s$ multiplies
   and divides along the input axis (column $j$ by $s_j$).
4. **Output error.**
   $$
   \text{loss}(\alpha) = \frac{1}{N \cdot \text{out}} \sum_{n, i} \Big( \big(X W^\top\big)_{n,i} - \big(X \hat W^\top\big)_{n,i} \Big)^2 ,
   \qquad X = \texttt{x\_calib}.
   $$

Return the $\alpha$ with the lowest loss and the normalized scales $s$ of step 2
that belong to it. If several grid entries reach the lowest loss, the one that comes
**first in `grid`** wins.

- `w`: float32, shape `(out, in)`; `in` is a multiple of `group_size`.
- `x_calib`: float32, shape `(N, in)`, $N \ge 1$. A column may be entirely zero (a
  dead channel).
- `bits`: a Python int, $2 \le \text{bits} \le 8$. `group_size`: a Python int
  $\ge 1$.
- `grid`: float32 tensor of shape `(G,)`, $G \ge 1$ distinct exponents in $[0, 1]$,
  in **any order**.
- Returns `(scales, alpha)`: `scales` is a float32 tensor of shape `(in,)`; `alpha`
  is a 0-d float32 tensor holding one of the values of `grid`.

Dividing column $j$ of the weight by $s_j$ is the same computation as dividing
activation $j$ by $s_j$: $X (Q(W s) / s)^\top = (X / s)\, Q(W s)^\top$. In a real
model the division is folded into the operation that produces $X$, and $Q(W s)$ is
what gets stored.

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

The channel magnitudes are $a = [1, 1, 16, 1]$: channel 2 is sixteen times larger
than the others. The row is one group, and 2 bits give $L = 3$. The layer's true
outputs are $X W^\top = [12.5,\ 11.9]$.

| $\alpha$ | $s$ (normalized) | $W \cdot s$ | $Q(W \cdot s)$ | $\hat W = Q / s$ | $X \hat W^\top$ | loss |
|---|---|---|---|---|---|---|
| $0$ | $1,\ 1,\ 1,\ 1$ | $-0.2,\ 0.5,\ 0.7,\ 1.0$ | $-0.2,\ 0.6,\ 0.6,\ 1.0$ | $-0.2,\ 0.6,\ 0.6,\ 1.0$ | $11.0,\ 10.2$ | $2.57$ |
| $0.5$ | $0.5,\ 0.5,\ 2,\ 0.5$ | $-0.1,\ 0.25,\ 1.4,\ 0.5$ | $-0.1,\ 0.4,\ 1.4,\ 0.4$ | $-0.2,\ 0.8,\ 0.7,\ 0.8$ | $12.6,\ 11.4$ | $0.13$ |
| $1$ | $0.25,\ 0.25,\ 4,\ 0.25$ | $-0.05,\ 0.125,\ 2.8,\ 0.25$ | $-0.05,\ -0.05,\ 2.8,\ -0.05$ | $-0.2,\ -0.2,\ 0.7,\ -0.2$ | $10.6,\ 11.4$ | $1.93$ |

Without scaling ($\alpha = 0$), the weight $0.7$ of the loud channel is rounded to
$0.6$. The error of $0.1$ looks small and is multiplied by an activation of $16$.
At $\alpha = 0.5$ the raw scales are $a^{0.5} = [1, 1, 4, 1]$, normalized by
$\sqrt{4 \cdot 1} = 2$. Scaling column 2 up makes it the maximum of its group, which
is represented exactly; the other three weights get a coarser grid and larger
errors, but their activations are small. At $\alpha = 1$ the scaled column dominates
the group so much that the step, $0.95$, swallows all three other weights: they
collapse onto the group minimum. The middle exponent wins.

## Constraints

- $1 \le \text{out} \le 64$, $1 \le \text{in} \le 256$, $1 \le N \le 256$,
  $G \le 20$. Entries of `w` and `x_calib` are finite with magnitude at most $10^3$.
- `alpha` must be exactly the expected grid entry. `scales` are compared to a
  relative tolerance of `1e-4` and an absolute tolerance of `1e-6`.
- Float32 rounding never decides a test. In every test, either the lowest loss is at
  least 2 percent below the loss of every other grid entry, or the test is about
  ties. Tie tests use activations that are all $\pm 1$, so every $a_j$ is exactly 1,
  every scale is exactly 1, and all grid entries have the same loss bit for bit.
- A Python loop over the grid is expected. No loops over channels, groups, or
  samples are needed.

## Why this matters

Rounding weights to the nearest grid point minimizes the error of the *weights*. A
model's quality depends on the error of its *outputs*, and the two differ whenever
input channels differ in magnitude: an error on a weight that multiplies a large
activation costs far more than the same error elsewhere. AWQ's observation is that
protecting about one percent of the channels, chosen by activation magnitude and not
by weight magnitude, recovers most of the accuracy lost at 3 and 4 bits, and that
the protection can be had without mixed precision by scaling those channels up
before quantization. The search is a few dozen forward passes of one layer on a
small calibration set, which is why AWQ checkpoints are a standard 4-bit format in
vLLM and llm-compressor, and why `mlx-lm` ships an AWQ quantizer. The classic bugs
all produce a plausible scale vector: an error measured on weights quietly returns
"no scaling", a signed activation mean turns into NaN only for channels that happen
to average below zero, and a scale applied to the weights without its inverse makes
every candidate look terrible except $\alpha = 0$.
