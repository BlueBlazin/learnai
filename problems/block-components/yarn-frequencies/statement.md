Implement YaRN, a method that extends a model trained on `original_max_positions`
tokens to handle `factor` times as many by rescaling its rotary frequencies. Fast
pairs keep their original frequency, slow pairs are divided by `factor`, and a linear
ramp over the pair index blends between them.

![This problem is RoPE, with its frequencies rescaled for a longer context.](assets/map.svg)

## Specification

```python
def yarn_inv_freq(
    dim, base, factor, original_max_positions, beta_fast=32.0, beta_slow=1.0, truncate=True
): ...
def yarn_tables(positions, inv_freq, factor): ...  # -> (cos, sin)
```

**`yarn_inv_freq`** computes the $\texttt{dim}/2$ inverse frequencies for a rotary
embedding of even width `dim`. Plain RoPE assigns pair $i$ the frequency

$$
\theta_i = \texttt{base}^{-2i/\texttt{dim}}, \qquad i = 0, \dots, \texttt{dim}/2 - 1 .
$$

Across the $L_0 = \texttt{original\_max\_positions}$ training positions, pair $i$
completes $L_0\,\theta_i / 2\pi$ full turns. Solving for the (fractional) pair index
that completes exactly $\beta$ turns gives

$$
\operatorname{boundary}(\beta) = \frac{\texttt{dim} \cdot \ln\!\big(L_0 / (2\pi\beta)\big)}{2 \ln \texttt{base}} .
$$

Compute the two ramp endpoints in this exact order:

1. Set $\text{low} = \operatorname{boundary}(\texttt{beta\_fast})$ and
   $\text{high} = \operatorname{boundary}(\texttt{beta\_slow})$.
2. If `truncate` is true, set $\text{low} \leftarrow \lfloor \text{low} \rfloor$ and
   $\text{high} \leftarrow \lceil \text{high} \rceil$. Otherwise, keep both fractional.
3. Clamp $\text{low} \leftarrow \max(\text{low}, 0)$ and
   $\text{high} \leftarrow \min(\text{high}, \texttt{dim} - 1)$. Note that the upper
   clamp is $\texttt{dim} - 1$, not the last pair index $\texttt{dim}/2 - 1$.

Then, for each pair index $i \in \{0, \dots, \texttt{dim}/2 - 1\}$, compute the linear
ramp and blended inverse frequency:

$$
\text{ramp}_i = \operatorname{clamp}\!\left(\frac{i - \text{low}}{\text{high} - \text{low}},\ 0,\ 1\right),
\qquad
\texttt{inv\_freq}_i = (1 - \text{ramp}_i)\,\theta_i + \text{ramp}_i\,\frac{\theta_i}{\texttt{factor}} .
$$

**`yarn_tables`** takes `T` token positions in `positions` and `P` inverse frequencies
in `inv_freq`, and returns a `(cos, sin)` tuple of cosine and sine tables scaled by a
shared attention factor $m$:

$$
\cos[t, i] = m \cos(\texttt{positions}[t] \cdot \texttt{inv\_freq}_i), \qquad
\sin[t, i] = m \sin(\texttt{positions}[t] \cdot \texttt{inv\_freq}_i), \qquad
m = 0.1 \ln(\texttt{factor}) + 1 ,
$$

for $0 \le t < T$ and $0 \le i < P$.

Parameter rules:

- `dim` is an even Python int.
- `base` is a Python float greater than 1 (called `rope_theta` in model configs).
- `factor` is a Python float with `factor >= 1`.
- `original_max_positions` is a Python int.
- `beta_fast` and `beta_slow` are Python floats with `beta_fast > beta_slow > 0`.
- `truncate` is a Python bool.
- `positions` contains non-negative integers in any order.
- `inv_freq` is any 1D float32 frequency tensor and need not come from `yarn_inv_freq`.

## Shapes

| name | shape | dtype |
|---|---|---|
| result of `yarn_inv_freq` | `(dim / 2,)` | float32 |
| `positions` | `(T,)` | integer |
| `inv_freq` | `(P,)` | float32 |
| `cos`, `sin` | `(T, P)` | float32 |

## Example

Consider a 16-wide embedding with `base = 10000`, trained on 512 positions and
extended by `factor = 8`. The plain RoPE frequencies are $\theta_i = 10^{-i/2}$.

```python
yarn_inv_freq(16, 10000.0, 8.0, 512)
# [1.0, 0.2470529, 0.05625, 0.01087033, 0.00125, 3.952847e-04, 1.25e-04, 3.952847e-05]

yarn_inv_freq(16, 10000.0, 8.0, 512, truncate=False)
# [1.0, 0.2989363, 0.06546508, 0.01151012, 0.00125, 3.952847e-04, 1.25e-04, 3.952847e-05]
```

The raw boundaries are $\operatorname{boundary}(32) = 16 \ln(512 / 64\pi) / (2 \ln
10^4) = 0.8119$ and $\operatorname{boundary}(1) = 16 \ln(512 / 2\pi) / (2 \ln 10^4) =
3.8222$.

| | low | high | $\text{ramp}_0, \dots, \text{ramp}_7$ | $\texttt{inv\_freq}_i / \theta_i$ |
|---|---|---|---|---|
| `truncate=True` | 0 | 4 | 0, 0.25, 0.5, 0.75, 1, 1, 1, 1 | 1, 0.78125, 0.5625, 0.34375, 0.125, 0.125, 0.125, 0.125 |
| `truncate=False` | 0.8119 | 3.8222 | 0, 0.0625, 0.3947, 0.7269, 1, 1, 1, 1 | 1, 0.9453, 0.6546, 0.3640, 0.125, 0.125, 0.125, 0.125 |

For pair 1 with truncation, $\theta_1 = 0.316228$ and $\text{ramp}_1 = 0.25$, giving
$0.75 \cdot 0.316228 + 0.25 \cdot 0.316228 / 8 = 0.247053$.

![The example with `truncate=True`, on a log scale. Pair 0 keeps $\theta_0$, and every pair from 4 on is divided by 8.](assets/frequencies.svg)

```python
cos, sin = yarn_tables([0, 1, 100], yarn_inv_freq(16, 10000.0, 8.0, 512), 8.0)
# cos[0] = [1.207944, 1.207944, ..., 1.207944]      m = 0.1 * ln(8) + 1 = 1.207944
# cos[1] = [0.652655, 1.171268, 1.206034, 1.207873, 1.207943, 1.207944, 1.207944, 1.207944]
# sin[1] = [1.016450, 0.295400, 0.067911, 0.013130, 0.001510, 0.000477, 0.000151, 0.000048]
```

## Constraints

- `dim` is even with $8 \le \texttt{dim} \le 64$.
- $100 \le \texttt{base} \le 160000$ and $1 \le \texttt{factor} \le 32$.
- $16 \le \texttt{original\_max\_positions} \le 2^{22}$ and
  $0.5 \le \texttt{beta\_slow} < \texttt{beta\_fast} \le 20000$.
- $0 \le \texttt{positions}[t] < 2^{18}$.
- In every test, after step 3, $\text{high} \ge 0$ and $\text{low} < \text{high}$.
- Frequencies are compared with relative tolerance `1e-4` and absolute tolerance
  `1e-12`, because the slow frequencies are very small and still matter. Pairs at or
  below `low` must equal $\theta_i$, and pairs at or above `high` must equal
  $\theta_i / \texttt{factor}$, both to relative tolerance `1e-5`.
- Tables are compared with relative tolerance `1e-4` and absolute tolerance
  $10^{-5} + 2 \cdot 10^{-7}\, p_{\max}$, where $p_{\max}$ is the largest position in
  the call. You may compute the angles in float32 or float64. Both are accepted since
  they differ by that order at large positions.
- For the parameters of gpt-oss-20b and DeepSeek-V4, and for a case where the upper
  clamp binds, expected values come from Hugging Face Transformers.

## Why this matters

gpt-oss-20b ships with `rope_type: "yarn"`, a factor of 32, and an original length of
4096 to support a 131072-token context. The frequencies are not stored in the
checkpoint. Instead, the inference engine recomputes them from six numbers in the
config. Using the wrong default for `truncate` alters 9 of that model's 32
frequencies, one of them by 76 percent, without raising an error. The model still
reads nearby tokens accurately while silently losing track of distant ones.
