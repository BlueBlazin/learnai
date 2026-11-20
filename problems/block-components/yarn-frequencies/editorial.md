## The idea

A rotary pair with frequency $\theta_i$ turns through $L_0\,\theta_i$ radians over the
$L_0$ positions of training. That number splits the head into two populations.

- **Fast pairs** wrap around many times. During training the model saw every angle
  of such a pair, at every offset, so positions beyond $L_0$ produce nothing new.
  These pairs carry fine relative distances ("three tokens back"), and slowing them
  down would blur exactly that. Leave them alone.
- **Slow pairs** complete less than one turn in $L_0$ positions. The model saw only
  an arc of their circle, and a position beyond $L_0$ puts them at angles it has never
  seen. Dividing their frequency by `factor` maps the new context $L_0 \cdot
  \texttt{factor}$ back onto the arc that was trained.

YaRN draws the line by the number of full turns, $L_0\,\theta_i / 2\pi$: at least
`beta_fast` (32) turns means fast, at most `beta_slow` (1) means slow. Because
$\theta_i = \texttt{base}^{-2i/\texttt{dim}}$ is monotone in $i$, each threshold is a
pair index, found by solving $L_0\,\texttt{base}^{-2i/\texttt{dim}} = 2\pi\beta$:

$$
i = \frac{\texttt{dim} \cdot \ln\!\big(L_0 / (2\pi\beta)\big)}{2 \ln \texttt{base}} .
$$

Between the two indices the implementation blends linearly in the pair index. The
second ingredient is unrelated to frequencies: with a longer context the attention
distribution flattens, and YaRN compensates by scaling the logits. It does so without
touching the attention code, by multiplying the cosine and sine tables by
$m = 0.1 \ln(\texttt{factor}) + 1$, which lengthens every rotated query and key by $m$
and every logit between rotated parts by $m^2$.

`base` is the argument name of the `rope` problem; configurations call it
`rope_theta`.

## Reference solution (PyTorch)

```python
def yarn_inv_freq(
    dim, base, factor, original_max_positions, beta_fast=32.0, beta_slow=1.0, truncate=True
):
    def boundary(rotations):
        turns = original_max_positions / (rotations * 2 * math.pi)
        return dim * math.log(turns) / (2 * math.log(base))

    low, high = boundary(beta_fast), boundary(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    index = torch.arange(dim // 2, dtype=torch.float32)
    ramp = torch.clamp((index - low) / (high - low), 0.0, 1.0)
    original = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    return (1.0 - ramp) * original + ramp * (original / factor)


def yarn_tables(positions, inv_freq, factor):
    angles = positions.to(torch.float32)[:, None] * inv_freq
    scale = 0.1 * math.log(factor) + 1.0
    return torch.cos(angles) * scale, torch.sin(angles) * scale
```

Written as `(1 - ramp) * original + ramp * (original / factor)`, the two ends are
exact in floating point: a ramp of 0 returns `original` and a ramp of 1 returns
`original / factor`, with no rounding from the blend. The bounds are ordinary Python
floats; only the ramp and the frequencies are tensors.

## Common mistakes

- **The ramp reversed,** or the two betas exchanged. Both slow down the fast pairs
  and leave the slow ones, the opposite of the method. Check the ends: pair 0 must
  have frequency exactly 1.
- **`truncate` ignored.** Always rounding, or never rounding, is right for one of
  the two shipped models and wrong for the other. The default is true.
- **The $2\pi$ dropped.** `beta` counts full turns. Without the $2\pi$ both bounds
  move toward slower pairs by $\texttt{dim} \ln(2\pi) / (2 \ln \texttt{base})$, about
  5 pairs for `dim = 64` and `base = 150000`.
- **The ramp over channel indices.** The exponent uses `arange(0, dim, 2)`; the ramp
  uses `arange(dim // 2)`. Reusing the first for the second doubles the slope.
- **Clamps.** The lower bound must be clamped at 0, or a short original context
  rescales pair 0. The upper bound is clamped at `dim - 1`. Clamping it at
  `dim / 2 - 1` looks more sensible and gives different frequencies whenever the
  ramp would end beyond the last pair. Leaving it unclamped is harder to notice. It
  changes a frequency only when the ramp starts before the last pair
  ($\text{low} < \texttt{dim}/2 - 1$) and would end beyond $\texttt{dim} - 1$, so
  the two unclamped bounds must be more than $\texttt{dim}/2$ apart. Their distance
  is $\texttt{dim} \ln(\texttt{beta\_fast} / \texttt{beta\_slow}) / (2 \ln
  \texttt{base})$, so that needs $\texttt{beta\_fast} / \texttt{beta\_slow} >
  \texttt{base}$: with the default 32 and 1, a base below 32. No shipped
  configuration read for this problem is near that, and the test that exercises the
  clamp uses `beta_fast = 20000` with `base = 100`.
- **Blending wavelengths.** Interpolating $1/\theta$ linearly gives the harmonic
  blend of the frequencies. It agrees at both ends and differs on the ramp.
- **Multiplying by the factor.** A longer context needs longer wavelengths: divide.
- **The table factor.** `log10` instead of `ln` (for a factor of 32, 1.15 instead of
  1.35), or the factor on the cosine table only, which makes the length of a rotated
  vector depend on its position.

## Complexity

$O(\texttt{dim})$ for the frequencies, computed once per model. $O(T \cdot
\texttt{dim})$ for the tables, usually precomputed for all positions.

## In the wild

Checked on 2026-11-21.

- **The paper.** "YaRN: Efficient Context Window Extension of Large Language Models"
  (arXiv 2309.00071), sections 3.2 and 3.3. It defines the ratio $r = L / \lambda$ of
  context length to wavelength, a ramp that is 0 below $\alpha$, 1 above $\beta$, and
  linear in $r$ between, the blend $(1 - \gamma)\,\theta / s + \gamma\,\theta$, the
  values $\alpha = 1$ and $\beta = 32$ for the Llama family, and
  $\sqrt{1/t} = 0.1 \ln(s) + 1$ for the scaling of the embeddings. Its footnote says
  that interpolating wavelengths instead "would work" too; the code does not do that.
- **The code.** `_compute_yarn_parameters` in `transformers/modeling_rope_utils.py`
  (`transformers` 5.23.0) is what this problem follows, including where it differs
  from the paper: its ramp is linear in the *pair index* between the two indices
  where $r$ equals `beta_fast` and `beta_slow`, not linear in $r$; it has the
  `truncate` option (default true), and the clamp of the upper bound to `dim - 1`. It
  also adds 0.001 to the upper bound when the two bounds coincide, a case this
  problem excludes.
- **gpt-oss-20b** (`config.json`): `rope_type: "yarn"`, factor 32, original length
  4096, `rope_theta` 150000, `truncate: false`, head width 64. The bounds are 8.09
  and 17.40, so pairs 0 to 8 are untouched, pairs 18 to 31 are divided by 32, and the
  tables carry $m = 1.3466$.
- **DeepSeek-V4** (`config.json`, `rope_scaling`): factor 16 from 65536, with
  `compress_rope_theta` 160000 on a rotary slice of 64 channels and no `truncate`
  key, so the bounds 15.45 and 24.71 become 15 and 25. Its configuration class in
  `transformers` sets `attention_factor` to 1 for these layers, with a comment that
  the model's reference code does not scale the tables: the frequencies are YaRN's,
  the table factor is not applied. A caller of `yarn_tables` would pass `factor = 1`
  there.

Both frequency vectors are in this problem's fixture, computed by the library through
each model's own rotary module (`fixtures/make.py`), together with the library's
result for the clamped parameters above.

The library forms the angle `position * inv_freq` in float32, where the spacing of
representable numbers near $10^5$ is about 0.008. A table built in float64 and cast
down is the more accurate one, and differs from the library's by up to that much at
the far end of the context. Both are accepted here; an engine that must reproduce a
reference bit for bit has to copy its precision as well as its formula.

The library also accepts an explicit `attention_factor`, and a pair `mscale`,
`mscale_all_dim` from which it derives one. Read all of them before assuming the
default.
