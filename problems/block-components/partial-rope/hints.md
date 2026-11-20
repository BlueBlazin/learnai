Before any trigonometry, write down for each function which channel indices rotate and
which index each of them is paired with. For `D = 16` and `factor = 0.5`, both
functions rotate eight channels. Are they the same eight? Is channel 0 paired with the
same partner?

---

Both functions are the half-split rotation of the `rope-layout-conversion` problem:
with first members `a` and second members `b`, the result is `a * cos - b * sin` and
`a * sin + b * cos`, written back where `a` and `b` came from.

- `partial_rope` slices out the first `r = int(D * factor)` channels, applies a
  complete half-split RoPE **of width `r`** to them (so `a = x[..., :r/2]`,
  `b = x[..., r/2:r]`, and the exponent is $-2i/r$), and concatenates `x[..., r:]`
  after the result.
- `proportional_rope` works on the whole head (`a = x[..., :D/2]`, `b = x[..., D/2:]`)
  with the exponent $-2i/D$. Not rotating a pair is the same as rotating it by the
  angle 0.

---

For `proportional_rope`, build a frequency vector of length `D/2`: the first
`n = int(factor * D // 2)` entries are $\texttt{base}^{-2i/D}$ for $i = 0, \dots, n-1$
(an `arange(0, 2 * n, 2)` divided by `D`, negated, as the exponent), and the remaining
`D/2 - n` entries are zeros. Then the code is the ordinary half-split RoPE: angles are
`positions[:, None] * freqs`, and a zero frequency gives $\cos = 1$, $\sin = 0$, which
returns the pair unchanged.

For `partial_rope`, the frequencies are an `arange(0, r, 2)` divided by `r` (not by
`D`), and the result is `concatenate((a * cos - b * sin, a * sin + b * cos, x[..., r:]))`
along the last axis.
