## The idea

A partial rotary embedding has to answer three questions, and the configuration key
`partial_rotary_factor` answers none of them by itself:

1. **Which channels rotate?**
2. **Which channel is paired with which?**
3. **Which frequencies do the rotated pairs get?**

The two functions are two consistent sets of answers.

`partial_rope` says: cut off the first $r = \texttt{int}(D \cdot \texttt{factor})$
channels and treat them as a head of width $r$. Everything a full RoPE would do to a
head of width $r$ happens inside the block: half-split pairs $(i, i + r/2)$, and the
frequencies $\texttt{base}^{-2i/r}$, which fall from 1 to nearly $1/\texttt{base}$.
The block has the complete range of wavelengths, just with fewer pairs to cover it.

`proportional_rope` says: keep the head's own half-split layout, pair $i$ with
$i + D/2$, and keep its frequency ladder $\texttt{base}^{-2i/D}$, then switch off all
but the first $n = \texttt{int}(\texttt{factor} \cdot D \mathbin{//} 2)$ pairs by
giving them frequency zero. The rotated pairs are the *fastest* $n$ rungs of the full
ladder, so the slowest rotated pair has the frequency
$\texttt{base}^{-2(n-1)/D} \approx \texttt{base}^{-\texttt{factor}}$, far above
$1/\texttt{base}$. With `factor = 0.25` and `base = 1e6` it is about $10^{-1.5}$:
the rotated part resolves distances up to a few hundred tokens and the unrotated part
is position-free.

The two readings are far apart on real numbers. For a head of width 512 with
`factor = 0.25` and `base = 1e6` (Gemma 4), the proportional form's slowest rotated
pair turns 0.033 radians per position; the default reading of the same three numbers
would rotate 128 channels and give its slowest pair 0.0000012. With `factor = 1` the
two functions coincide: both are the half-split RoPE of the whole head.

`base` is the argument name of the `rope` problem; configurations call it
`rope_theta`.

Both keep what makes RoPE work. Each pair is rotated by an angle proportional to the
position, the same for queries and keys, so the dot product of a rotated query at
position $m$ and a rotated key at position $n$ depends on $m - n$ only, and the
channels that are not rotated contribute a term with no position in it at all.

## Reference solution (PyTorch)

```python
def partial_rope(x, positions, base, factor):
    dim = x.shape[-1]
    r = int(dim * factor)
    half = r // 2
    freqs = base ** (-torch.arange(0, r, 2, dtype=torch.float32) / r)
    angles = positions.to(torch.float32)[:, None] * freqs  # (T, r/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    a, b = x[..., :half], x[..., half:r]  # pair i is (i, i + r/2)
    return torch.cat((a * cos - b * sin, a * sin + b * cos, x[..., r:]), dim=-1)


def proportional_rope(x, positions, base, factor):
    dim = x.shape[-1]
    half = dim // 2
    n_rot = int(factor * dim // 2)
    freqs = base ** (-torch.arange(0, 2 * n_rot, 2, dtype=torch.float32) / dim)
    freqs = torch.cat((freqs, torch.zeros(half - n_rot)))  # frequency 0: not rotated
    angles = positions.to(torch.float32)[:, None] * freqs  # (T, D/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    a, b = x[..., :half], x[..., half:]  # pair i is (i, i + D/2)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)
```

The zero frequencies make the unrotated pairs fall out of the same formula: the angle
is 0, the cosine 1, the sine 0. That is also how the library builds it, and it is why
the proportional form returns cosine and sine tables as wide as the whole head while
the default form returns tables of width $r$.

## Common mistakes

- **The denominators swapped.** $\texttt{base}^{-2i/D}$ in `partial_rope`, or
  $\texttt{base}^{-2i/(2n)}$ in `proportional_rope`. Pair 0 has frequency 1 either
  way, so the first channel of the output is right and the error grows along the
  head. This is the mistake the problem exists for: it is what you get by reading one
  model's code and another model's configuration.
- **The pairing swapped.** Pairing $i$ with $i + n$ inside the first $2n$ channels in
  the proportional form rotates the wrong half of the channels: channels $n$ to
  $2n - 1$ instead of $D/2$ to $D/2 + n - 1$.
- **Rotating the last $r$ channels.** Some architectures do put the rotated slice at
  the end of the head (DeepSeek-V4 does, see `shared-kv-rotation`); the default
  convention of this problem rotates the first $r$.
- **Adjacent pairs.** $(0, 1), (2, 3), \dots$ inside the block is the interleaved
  layout of the `rope` problem. It is a valid RoPE and a different function of `x`.
- **Ignoring the factor.** Rotating all $D$ channels is what a loader does when it
  does not know the key. The scores still depend only on relative position, so
  nothing looks broken.
- **Rounding the pair count.** $\texttt{int}(\texttt{factor} \cdot D \mathbin{//} 2)$
  floors. For $D = 12$ and `factor = 0.25` it is 1 pair, not 2. Both counts are
  float expressions in the shipped code, so a factor such as 0.58 with $D = 200$
  gives `int(115.99999999999999) = 115`, an odd "width". This problem keeps to
  factors that are multiples of 1/8, for which the products are exact; a loader
  should check the count it gets instead of trusting the arithmetic.
- **Ignoring the positions,** or rotating clockwise: the two mistakes of the `rope`
  problem, both invisible to a test that only looks at relative-position properties
  of a prompt that starts at 0.

## Complexity

$O(B\,H\,T\,D)$ time and memory. Only the `(T, r/2)` or `(T, D/2)` angle table needs
transcendental functions; engines precompute it per position.

## In the wild

Checked on 2026-11-21 against the code installed with `transformers` 5.23.0 and the
`config.json` of each checkpoint on the Hugging Face Hub.

- **Default with a factor.** `Qwen/Qwen3.5-0.8B` has `rope_type: "default"`,
  `partial_rotary_factor: 0.25`, `rope_theta: 1e7`, and a head width of 256.
  `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters` computes
  `dim = int(head_dim * partial_rotary_factor)` and
  `1 / base ** (arange(0, dim, 2) / dim)`, and `apply_rotary_pos_emb` in
  `modeling_qwen3_5.py` rotates `q[..., :rotary_dim]` with `rotate_half` and
  concatenates the rest. That is `partial_rope`: 64 rotated channels with the ladder
  of a 64-wide head. (For images the model assigns three position ids per token; for
  text they are equal and the code reduces to this.)
- **Proportional.** `google/gemma-4-E2B-it` has, for its full-attention layers,
  `rope_type: "proportional"`, `partial_rotary_factor: 0.25`, `rope_theta: 1e6`, and
  a head width of 512. `_compute_proportional_rope_parameters` in
  `modeling_rope_utils.py` computes `rope_angles = int(rope_proportion * head_dim // 2)`,
  the frequencies `1 / base ** (arange(0, 2 * rope_angles, 2) / head_dim)`, and pads
  with zeros to `head_dim // 2`; `apply_rotary_pos_emb` in `modeling_gemma4.py` then
  applies `rotate_half` over the whole head. That is `proportional_rope`. Its
  docstring states the difference: unlike the other RoPE functions that read this
  parameter, it "will always return an encoding that is the size of `head_dim`".
  Gemma 4's sliding-window layers use plain RoPE with `rope_theta: 1e4` and a head
  width of 256, so one model carries two position encodings.

The public test "Qwen3.5 and Gemma 4" compares your functions with the outputs of
those two code paths (`fixtures/make.py` regenerates them).

The key can also be read and then not used. The `config.json` of `LiquidAI/LFM3-1.7B`
gives its full-attention layers `rope_type: "default"` with
`partial_rotary_factor: 0.5`, while the configuration class in `transformers` 5.23.0
defaults to `rope_type: "proportional"` with the same factor, and the default path of
`Lfm3RotaryEmbedding` does not read the factor at all: run with the shipped values,
the installed code builds 64 nonzero frequencies for a 128-wide head, a full rotation.
Which of the three the weights were trained with cannot be decided from the
configuration, which is the practical lesson: when you port a model, compare the
frequency table and one rotated vector with the reference implementation before you
compare anything else.
