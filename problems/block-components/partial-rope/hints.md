Before writing any trigonometry, list which channel indices rotate and which partner
each index pairs with in each function. For `D = 16` and `factor = 0.5`, both
functions rotate eight channels. Are they the same eight channels? Is channel 0 paired
with the same partner?

---

Both functions use the half-split rotation from `rope-layout-conversion`. Given first
members `a` and second members `b`, the rotated outputs are `a * cos - b * sin` and
`a * sin + b * cos`, written back to the channels where `a` and `b` came from.

- `partial_rope` slices out the first `r = int(D * factor)` channels, applies a
  complete half-split RoPE **of width `r`** (`a = x[..., :r/2]`, `b = x[..., r/2:r]`,
  exponent $-2i/r$), and concatenates `x[..., r:]` at the end.
- `proportional_rope` operates on the whole head (`a = x[..., :D/2]`,
  `b = x[..., D/2:]`) with exponent $-2i/D$. Leaving a pair unrotated is equivalent to
  rotating it by angle 0.

---

For `proportional_rope`, build a frequency vector `freqs` of length `D/2`. The first
`n = int(factor * D // 2)` entries are $\texttt{base}^{-2i/D}$ for $i = 0, \dots, n-1$
(using `arange(0, 2 * n, 2)` divided by `D`, negated, as the exponent), and the
remaining `D/2 - n` entries are zeros. Then run standard half-split RoPE with
`angles = positions[:, None] * freqs`. Every zero frequency yields $\cos = 1$ and
$\sin = 0$, leaving that pair unchanged.

For `partial_rope`, build `freqs` using `arange(0, r, 2)` divided by `r` (not `D`),
and return `concatenate((a * cos - b * sin, a * sin + b * cos, x[..., r:]))` along the
last axis.
