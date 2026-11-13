## The idea

Think of each adjacent pair $(x_{2i}, x_{2i+1})$ as a point in the plane, or
equivalently as the complex number $z_i = x_{2i} + \mathrm{i}\,x_{2i+1}$. RoPE
multiplies it by $e^{\mathrm{i} p \theta_i}$: a rotation by an angle proportional to
the position $p$, at a frequency $\theta_i = \text{base}^{-2i/D}$ that falls
geometrically from 1 for the first pair to about $1/\text{base}$ for the last.

Why this encodes *relative* position: the dot product of two real vectors is the real
part of $\sum_i z^q_i\,\overline{z^k_i}$. After rotating the query for position $m$ and
the key for position $n$,

$$
\sum_i \big(z^q_i e^{\mathrm{i} m \theta_i}\big)\,\overline{\big(z^k_i e^{\mathrm{i} n \theta_i}\big)}
= \sum_i z^q_i\,\overline{z^k_i}\; e^{\mathrm{i}(m - n)\theta_i} .
$$

Only $m - n$ survives. Rotations also preserve length, so RoPE changes no norms. Both
facts are tested as properties, and both hold for any consistent pairing and either
direction of rotation, which is exactly why a wrong convention is so hard to notice:
a model's attention still "works", just not with the weights it was trained with.

Implementation is then three steps: build the `(T, D/2)` table of angles from the
positions and frequencies, rotate the even-indexed and odd-indexed halves of `x`, and
interleave the results.

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
right, so batch and head axes need no handling. Stacking on a new last axis and
reshaping puts each rotated pair back into its two adjacent slots. The JAX and MLX
versions are direct translations.

## Common mistakes

- **Half-split pairing.** Pairing dimension $i$ with $i + D/2$ (`x[..., :D/2]` and
  `x[..., D/2:]`, the `rotate_half` idiom) is the GPT-NeoX and Hugging Face
  convention. It is a valid RoPE and a different function: it equals the interleaved
  one only after permuting the dimensions of `x`, which for a real model means
  permuting rows of the query and key projection weights.
- **Rotating the right pairs and concatenating the halves.** Computing with `0::2` and
  `1::2` and then joining with `cat` mixes the two conventions: the output is in
  half-split order.
- **Wrong frequency exponent.** $\text{base}^{-i/D}$ (the factor 2 forgotten) and
  $\text{base}^{-2(i+1)/D}$ (the paper's 1-based index $\theta_i =
  10000^{-2(i-1)/d}$ copied into 0-based code with the wrong shift) both give smooth,
  plausible frequency tables. Check the ends: $\theta_0$ must be exactly 1.
- **Ignoring the positions.** Using `arange(T)` is right for a full prompt starting at
  0 and wrong for every cached decoding step, where `T = 1` and the position is the
  current length of the sequence.
- **Rotating clockwise.** Negating the angle keeps the relative-position property
  (scores then depend on $n - m$ with the opposite sign), so no property test catches
  it. It is still a different function from the one the weights expect.
- **A sign error in one component.** `x1 * sin - x2 * cos` turns the rotation into a
  reflection. Norms are still preserved, but position 0 is no longer the identity and
  the relative-position property is lost.

## Complexity

$O(B\,H\,T\,D)$ time and memory: one multiply-add pair per element. The `(T, D/2)`
table of cosines and sines is the only part that involves transcendental functions.

## In the wild

Engines precompute the cosine and sine tables once for all positions up to the context
length and index them by position, so the per-step cost is a gather and a few
multiplies. Meta's reference code stores the table as complex numbers
(`torch.polar(ones, angles)`), views the last axis of `x` as `(D/2, 2)` complex pairs,
and multiplies, which is the interleaved convention by construction. Hugging Face's
Llama concatenates the frequency vector with itself and uses `rotate_half`, the
half-split convention. Both run the same published models correctly, because the
converted checkpoints are laid out for the convention of the code that loads them. A
port from one code base to another has to carry the convention along with the weights.

Precision deserves a thought at long context. The angle $p\,\theta_i$ is computed in
float32, whose spacing near $10^5$ is about $0.008$, so at positions in the hundreds of
thousands the fastest pairs have noticeably quantized angles. That is the reason for
the looser tolerance at large positions in this problem, and the reason careful
implementations build the table in float64 and cast the cosines and sines down.

RoPE is applied to queries and keys only, never to values, and after the projection
into heads. With a KV cache, the cached keys are stored already rotated: each key was
rotated once, for its own absolute position, and thanks to the relative-position
property it never has to be touched again as the sequence grows.
