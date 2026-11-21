Implement YaRN, which lets a model trained on `original_max_positions` tokens run on
`factor` times as many by rescaling its rotary frequencies: fast pairs keep their
frequency, slow pairs are divided by `factor`, and a linear ramp over the pair index
blends the two.

![This problem is RoPE, with its frequencies rescaled for a longer context.](assets/map.svg)

## Specification

```python
def yarn_inv_freq(
    dim, base, factor, original_max_positions, beta_fast=32.0, beta_slow=1.0, truncate=True
): ...
def yarn_tables(positions, inv_freq, factor): ...  # -> (cos, sin)
```

**`yarn_inv_freq`** returns the $\texttt{dim}/2$ inverse frequencies of a rotary
embedding of width `dim`. Plain RoPE gives pair $i$ the frequency

$$
\theta_i = \texttt{base}^{-2i/\texttt{dim}}, \qquad i = 0, \dots, \texttt{dim}/2 - 1 .
$$

Over the $L_0 = \texttt{original\_max\_positions}$ positions of training, pair $i$
makes $L_0\,\theta_i / 2\pi$ full turns. The (fractional) pair index that makes exactly
$\beta$ turns is

$$
\operatorname{boundary}(\beta) = \frac{\texttt{dim} \cdot \ln\!\big(L_0 / (2\pi\beta)\big)}{2 \ln \texttt{base}} .
$$

The two ends of the ramp are computed in this order:

1. $\text{low} = \operatorname{boundary}(\texttt{beta\_fast})$ and
   $\text{high} = \operatorname{boundary}(\texttt{beta\_slow})$.
2. If `truncate` is true, $\text{low} \leftarrow \lfloor \text{low} \rfloor$ and
   $\text{high} \leftarrow \lceil \text{high} \rceil$; otherwise both stay fractional.
3. $\text{low} \leftarrow \max(\text{low}, 0)$ and
   $\text{high} \leftarrow \min(\text{high}, \texttt{dim} - 1)$. The upper clamp is
   $\texttt{dim} - 1$, not the last pair index.

Then, for every pair index $i$,

$$
\text{ramp}_i = \operatorname{clamp}\!\left(\frac{i - \text{low}}{\text{high} - \text{low}},\ 0,\ 1\right),
\qquad
\texttt{inv\_freq}_i = (1 - \text{ramp}_i)\,\theta_i + \text{ramp}_i\,\frac{\theta_i}{\texttt{factor}} .
$$

**`yarn_tables`** returns the cosine and sine tables for the given positions, both
multiplied by the same number:

$$
\cos[t, i] = m \cos(\texttt{positions}[t] \cdot \texttt{inv\_freq}_i), \qquad
\sin[t, i] = m \sin(\texttt{positions}[t] \cdot \texttt{inv\_freq}_i), \qquad
m = 0.1 \ln(\texttt{factor}) + 1 .
$$

Arguments and results:

- `dim`: an even Python int. `base`: a Python float greater than 1 (`rope_theta` in
  configurations). `factor`: a Python float, at least 1. `original_max_positions`: a
  Python int.
- `beta_fast`, `beta_slow`: Python floats with `beta_fast > beta_slow > 0`.
- `truncate`: a Python bool.
- `yarn_inv_freq` returns a float32 tensor of shape `(dim / 2,)`.
- `positions`: integer tensor, shape `(T,)`, values $\ge 0$ in any order.
- `inv_freq`: float32 tensor, shape `(P,)`; it need not come from `yarn_inv_freq`.
- `yarn_tables` returns a tuple `(cos, sin)` of float32 tensors of shape `(T, P)`.

## Shapes

| name | shape | dtype |
|---|---|---|
| result of `yarn_inv_freq` | `(dim / 2,)` | float32 |
| `positions` | `(T,)` | integer |
| `inv_freq` | `(P,)` | float32 |
| `cos`, `sin` | `(T, P)` | float32 |

## Example

A 16-wide embedding with `base = 10000`, trained on 512 positions and extended by
`factor = 8`. The plain frequencies are $\theta_i = 10^{-i/2}$.

```python
yarn_inv_freq(16, 10000.0, 8.0, 512)
# [1.0, 0.2470529, 0.05625, 0.01087033, 0.00125, 3.952847e-04, 1.25e-04, 3.952847e-05]

yarn_inv_freq(16, 10000.0, 8.0, 512, truncate=False)
# [1.0, 0.2989363, 0.06546508, 0.01151012, 0.00125, 3.952847e-04, 1.25e-04, 3.952847e-05]
```

$\operatorname{boundary}(32) = 16 \ln(512 / 64\pi) / (2 \ln 10^4) = 0.8119$ and
$\operatorname{boundary}(1) = 16 \ln(512 / 2\pi) / (2 \ln 10^4) = 3.8222$.

| | low | high | $\text{ramp}_0, \dots, \text{ramp}_7$ | $\texttt{inv\_freq}_i / \theta_i$ |
|---|---|---|---|---|
| `truncate=True` | 0 | 4 | 0, 0.25, 0.5, 0.75, 1, 1, 1, 1 | 1, 0.78125, 0.5625, 0.34375, 0.125, 0.125, 0.125, 0.125 |
| `truncate=False` | 0.8119 | 3.8222 | 0, 0.0625, 0.3947, 0.7269, 1, 1, 1, 1 | 1, 0.9453, 0.6546, 0.3640, 0.125, 0.125, 0.125, 0.125 |

For pair 1 with truncation: $0.75 \cdot 0.316228 + 0.25 \cdot 0.316228 / 8 = 0.247053$.

![The example with `truncate=True`, on a log scale. Pair 0 keeps $\theta_0$, and every pair from 4 on is divided by 8.](assets/frequencies.svg)

```python
cos, sin = yarn_tables([0, 1, 100], yarn_inv_freq(16, 10000.0, 8.0, 512), 8.0)
# cos[0] = [1.207944, 1.207944, ..., 1.207944]      m = 0.1 * ln(8) + 1 = 1.207944
# cos[1] = [0.652655, 1.171268, 1.206034, 1.207873, 1.207943, 1.207944, 1.207944, 1.207944]
# sin[1] = [1.016450, 0.295400, 0.067911, 0.013130, 0.001510, 0.000477, 0.000151, 0.000048]
```

## Constraints

- `dim` is even, $8 \le \texttt{dim} \le 64$; $100 \le \texttt{base} \le 160000$;
  $1 \le \texttt{factor} \le 32$; $16 \le \texttt{original\_max\_positions} \le 2^{22}$;
  $0.5 \le \texttt{beta\_slow} < \texttt{beta\_fast} \le 20000$;
  $0 \le$ `positions[t]` $< 2^{18}$.
- In every test, after step 3, $\text{high} \ge 0$ and $\text{low} < \text{high}$.
- Frequencies are compared to a relative tolerance of `1e-4` and an absolute tolerance
  of `1e-12`: the slow frequencies are tiny, and they count. Pairs at or below `low`
  must equal $\theta_i$ and pairs at or above `high` must equal
  $\theta_i / \texttt{factor}$ to a relative tolerance of `1e-5`.
- Tables are compared to a relative tolerance of `1e-4` and an absolute tolerance of
  $10^{-5} + 2 \cdot 10^{-7}\, p_{\max}$, where $p_{\max}$ is the largest position in
  the call. The angle may be formed in float32 or in float64; the two differ by that
  order at large positions, and both are accepted.
- For the parameters of gpt-oss-20b and DeepSeek-V4, and for one set that makes the
  upper clamp bind, the expected values are those computed by Hugging Face
  Transformers.

## Why this matters

gpt-oss-20b ships with `rope_type: "yarn"`, a factor of 32, and an original length of
4096 for a context of 131072 tokens. The frequencies are not in the checkpoint: the
engine recomputes them from six numbers in the configuration. Reading `truncate` with
the wrong default changes 9 of that model's 32 frequencies, one of them by 76 percent,
and no error is raised; the model reads nearby tokens correctly and loses track of
distant ones.
