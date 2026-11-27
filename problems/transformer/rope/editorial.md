## The idea

Treat each adjacent pair $(x_{2i}, x_{2i+1})$ as a point in the plane, or equivalently
as the complex number $z_i = x_{2i} + \mathrm{i}\,x_{2i+1}$. RoPE multiplies each
complex number by $e^{\mathrm{i} p \theta_i}$, rotating it by an angle proportional to
position $p$. The frequencies $\theta_i = \text{base}^{-2i/D}$ decay geometrically from
1 for the first pair to roughly $1/\text{base}$ for the last.

To see why this encodes *relative* position, write the real dot product of two vectors
as the real part of $\sum_i z^q_i\,\overline{z^k_i}$. After rotating the query for
position $m$ and the key for position $n$,

$$
\sum_i \big(z^q_i e^{\mathrm{i} m \theta_i}\big)\,\overline{\big(z^k_i e^{\mathrm{i} n \theta_i}\big)}
= \sum_i z^q_i\,\overline{z^k_i}\; e^{\mathrm{i}(m - n)\theta_i} .
$$

Only the offset $m - n$ survives. Because rotations also preserve length, RoPE leaves
vector norms unchanged. Both properties hold for any consistent pairing and either
rotation direction. That is why a mismatched convention is easy to miss. Attention still
behaves smoothly, just not in the coordinate system the weights were trained for.

The implementation has three steps. First, build the `(T, D/2)` table of angles from
`positions` and $\theta$. Next, rotate the even-indexed and odd-indexed halves of `x`.
Finally, interleave the rotated pairs back into the original layout.

## Reference solution (PyTorch)

```python
def apply_rope(x, positions, base=10000.0):
    dim = x.shape[-1]
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = positions.to(torch.float32)[:, None] * theta  # (T, D/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]  # (B, H, T, D/2)
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.reshape(x.shape)  # interleave
```

The angle table has shape `(T, D/2)` and broadcasts against `(B, H, T, D/2)` from the
right, so the batch and head axes need no special handling. Stacking on a new last axis
and reshaping back to `x.shape` places each rotated pair into adjacent slots. The JAX
and MLX solutions follow the same structure.

## Common mistakes

- **Half-split pairing.** Pairing dimension $i$ with $i + D/2$ (`x[..., :D/2]` and
  `x[..., D/2:]`, the `rotate_half` pattern) is the GPT-NeoX and Hugging Face
  convention. It is a valid RoPE, but a different function of `x`. It matches the
  interleaved convention only after permuting the dimensions of `x` (and the
  corresponding rows of the query and key weights).
- **Rotating the right pairs and concatenating the halves.** Slicing with `0::2` and
  `1::2` and then joining with `cat` mixes the two conventions and writes the output in
  half-split order.
- **Wrong frequency exponent.** Computing $\text{base}^{-i/D}$ (forgetting the factor
  of 2) or $\text{base}^{-2(i+1)/D}$ (copying the paper's 1-based formula
  $\theta_i = 10000^{-2(i-1)/d}$ into 0-based code with the wrong shift) still produces
  a smooth frequency table. Always check the first entry, which must satisfy
  $\theta_0 = 1$.
- **Ignoring `positions`.** Using `arange(T)` works for a full prompt starting at 0,
  but fails on every cached decoding step where `T = 1` and the token sits at the end
  of the sequence.
- **Rotating clockwise.** Negating the angle preserves the relative-position property
  (scores still depend on $n - m$, just with the opposite sign), so property tests will
  not catch it. It is still a different function from what the weights expect.
- **A sign error in one component.** Writing `x1 * sin - x2 * cos` turns the rotation
  into a reflection. Vector norms are still preserved, but position 0 is no longer the
  identity and the relative-position property breaks.

## Complexity

Time and memory are both $O(B\,H\,T\,D)$, with one multiply-add pair per element. Only
the `(T, D/2)` table of cosines and sines requires trigonometric operations.

## In the wild

Inference engines precompute the cosine and sine tables once for all positions up to
the maximum context length and index into them by position. Each step then costs only a
gather and a few elementwise multiplies. Meta's reference code stores the table as
complex numbers (`torch.polar(ones, angles)`), views the last axis of `x` as `(D/2, 2)`
complex pairs, and multiplies directly, which enforces the interleaved convention by
construction. Hugging Face's Llama implementation concatenates the frequency vector with
itself and uses `rotate_half` (the half-split convention). Both run the same published
models accurately because each checkpoint is converted to match the convention of the
loader. Porting weights between codebases requires carrying the pairing convention
along with them.

Numerical precision matters at long context lengths. Computing $p\,\theta_i$ in float32
becomes noticeably quantized at positions in the hundreds of thousands, since float32
spacing near $10^5$ is about $0.008$. That is why this problem uses a looser tolerance
at large positions, and why production engines often compute the angle table in float64
before casting the cosines and sines down.

RoPE is applied only to queries and keys after the head projection, never to values.
With a KV cache, keys are stored already rotated for their own absolute positions.
Thanks to the relative-position property, cached keys never need to be re-rotated as
the sequence grows.
