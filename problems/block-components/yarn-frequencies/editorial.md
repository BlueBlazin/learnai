## The idea

A rotary pair with frequency $\theta_i$ sweeps through $L_0\,\theta_i$ radians over
the $L_0$ positions of training. That total rotation splits the head into two groups.

- **Fast pairs** wrap around many times. During training, the model saw every angle of
  these pairs at every offset, so positions beyond $L_0$ introduce nothing unfamiliar.
  These pairs encode fine local distances (such as "three tokens back"), and slowing
  them down blurs that resolution. YaRN leaves them untouched.
- **Slow pairs** complete less than one full turn over $L_0$ positions. The model only
  ever saw a partial arc of their circle, so positions beyond $L_0$ push them into
  unseen angles. Dividing their frequency by `factor` maps the extended context
  $L_0 \cdot \texttt{factor}$ back onto the trained arc.

YaRN separates these groups by the number of full turns, $L_0\,\theta_i / 2\pi$. Pairs
making at least `beta_fast` (32) turns are treated as fast, and pairs making at most
`beta_slow` (1) turn are treated as slow. Because
$\theta_i = \texttt{base}^{-2i/\texttt{dim}}$ decreases monotonically with $i$, each
threshold corresponds to a single pair index found by solving
$L_0\,\texttt{base}^{-2i/\texttt{dim}} = 2\pi\beta$:

$$
i = \frac{\texttt{dim} \cdot \ln\!\big(L_0 / (2\pi\beta)\big)}{2 \ln \texttt{base}} .
$$

Between those two indices, the implementation blends linearly in the pair index.

The second piece addresses attention entropy rather than frequencies. At longer context
lengths, the attention distribution tends to flatten, so YaRN sharpens the logits. To
do this without changing the attention kernel, it multiplies both the cosine and sine
tables by $m = 0.1 \ln(\texttt{factor}) + 1$. That scales every rotated query and key
by $m$, which scales their dot product by $m^2$. (In model configs, `base` is called
`rope_theta`.)

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

Writing the blend as `(1 - ramp) * original + ramp * (original / factor)` keeps both
endpoints exact in floating point. A ramp value of 0 returns `original` directly, and
a ramp value of 1 returns `original / factor` with no blend rounding. The bounds stay
plain Python floats, while only the ramp and frequencies are tensors.

## Common mistakes

- **Reversing the ramp or swapping the betas.** Both mistakes slow down the fast pairs
  and leave the slow ones unchanged, which is the opposite of YaRN. Always verify that
  pair 0 has frequency 1.
- **Ignoring `truncate`.** Always rounding or never rounding matches one shipped model
  and breaks the other. The default is `True`.
- **Dropping the $2\pi$ factor.** `beta` counts full turns, not radians. Omitting
  $2\pi$ shifts both boundaries toward slower pairs by
  $\texttt{dim} \ln(2\pi) / (2 \ln \texttt{base})$, or about 5 pairs for `dim = 64` and
  `base = 150000`.
- **Evaluating the ramp over channel indices.** The frequency exponent uses
  `arange(0, dim, 2)`, whereas the ramp uses `arange(dim // 2)`. Reusing the channel
  indices doubles the ramp slope.
- **Wrong boundary clamps.** The lower bound must be clamped at 0, or a short original
  context rescales pair 0. The upper bound is clamped at `dim - 1`. Clamping at
  `dim / 2 - 1` looks more natural, yet it changes the frequencies whenever the ramp
  extends past the last pair. Leaving the upper bound unclamped is harder to spot
  because it only changes a frequency when $\text{low} < \texttt{dim}/2 - 1$ and the
  unclamped upper bound exceeds $\texttt{dim} - 1$. That requires the two unclamped
  bounds to be more than $\texttt{dim}/2$ apart, which happens when
  $\texttt{beta\_fast} / \texttt{beta\_slow} > \texttt{base}$ (for default 32 and 1, a
  base below 32). No shipped config read here hits that regime, so the test suite
  exercises the clamp using `beta_fast = 20000` and `base = 100`.
- **Blending wavelengths instead of frequencies.** Linearly interpolating $1/\theta$
  produces the harmonic blend of the frequencies. It matches at both ends and differs
  across the ramp.
- **Multiplying by `factor` instead of dividing.** Extending context requires longer
  wavelengths, which means smaller frequencies.
- **Wrong table scaling.** Using `log10` instead of `ln` gives 1.15 instead of 1.35 at
  `factor = 32`. Scaling only the cosine table makes the norm of a rotated vector
  depend on its position.

## Complexity

`yarn_inv_freq` takes $O(\texttt{dim})$ time and memory and runs once per model.
`yarn_tables` takes $O(T \cdot \texttt{dim})$ time and memory for `T` positions, and
engines usually precompute the tables.

## In the wild

Checked on 2026-11-21.

- **The paper.** "YaRN: Efficient Context Window Extension of Large Language Models"
  (arXiv 2309.00071), sections 3.2 and 3.3, defines the ratio $r = L / \lambda$ of
  context length to wavelength. Its ramp $\gamma$ is 0 below $\alpha$, 1 above $\beta$,
  and linear in $r$ between them, using defaults $\alpha = 1$ and $\beta = 32$ for
  Llama. For scale factor $s$, it blends frequencies as
  $(1 - \gamma)\,\theta / s + \gamma\,\theta$ and scales the embeddings by
  $\sqrt{1/t} = 0.1 \ln(s) + 1$. A footnote mentions that interpolating wavelengths
  "would work" too, though the code does not do that.
- **The code.** This problem follows `_compute_yarn_parameters` in
  `transformers/modeling_rope_utils.py` (`transformers` 5.23.0), including its
  differences from the paper. Its ramp is linear in the *pair index* between the
  indices where $r$ equals `beta_fast` and `beta_slow` rather than linear in $r$. It
  also adds the `truncate` option (default `True`), clamps the upper bound to
  `dim - 1`, and adds 0.001 to the upper bound when the two bounds coincide (a case
  excluded here).
- **gpt-oss-20b.** Its `config.json` sets `rope_type: "yarn"`, factor 32, original
  length 4096, `rope_theta` 150000, `truncate: false`, and head width 64. The bounds
  are 8.09 and 17.40, so pairs 0 to 8 stay untouched, pairs 18 to 31 are divided by
  32, and the tables use $m = 1.3466$.
- **DeepSeek-V4.** Its `config.json` (`rope_scaling`) uses factor 16 from 65536 with
  `compress_rope_theta` 160000 on a 64-channel rotary slice and no `truncate` key, so
  the bounds 15.45 and 24.71 round to 15 and 25. Its config class in `transformers`
  sets `attention_factor` to 1 for these layers, noting that the reference code does
  not scale the tables. The frequencies follow YaRN while the table scale is omitted,
  which corresponds to passing `factor = 1` to `yarn_tables`.

Both frequency vectors are stored in this problem's fixture, computed by `transformers`
through each model's rotary module (`fixtures/make.py`), along with the clamped test
case.

The library forms `position * inv_freq` in float32, where the spacing between adjacent
floats near $10^5$ is about 0.008. Computing the angle in float64 and casting down is
more accurate, yet it differs from the library by up to that amount at the far end of
the context. Both are accepted here. Any engine aiming for bit-for-bit parity with a
reference must match its precision as well as its formula.

The library also accepts an explicit `attention_factor`, as well as `mscale` and
`mscale_all_dim` parameters that can derive one. Check all of them before assuming the
default formula.
