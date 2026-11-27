## The idea

A partial rotary embedding has to answer three questions, and the config key
`partial_rotary_factor` answers none of them by itself:

1. **Which channels rotate?**
2. **Which channel is paired with which?**
3. **Which frequencies do the rotated pairs get?**

The two functions represent two consistent sets of answers.

`partial_rope` slices off the first $r = \texttt{int}(D \cdot \texttt{factor})$
channels and treats them as a standalone head of width $r$. Every detail of a full
RoPE of width $r$ applies inside that slice. Channel $i$ pairs with $i + r/2$, and its
frequency is $\texttt{base}^{-2i/r}$, falling from 1 to nearly $1/\texttt{base}$. The
slice keeps the full range of wavelengths, using fewer pairs to span it.

`proportional_rope` keeps the full head's half-split layout and frequency ladder.
Channel $i$ pairs with $i + D/2$ at frequency $\texttt{base}^{-2i/D}$, and all pairs
past the first $n = \texttt{int}(\texttt{factor} \cdot D \mathbin{//} 2)$ are turned
off by setting their frequency to zero. Because the rotated pairs are the *fastest*
$n$ rungs of the ladder, the slowest rotated pair has frequency
$\texttt{base}^{-2(n-1)/D} \approx \texttt{base}^{-\texttt{factor}}$, which is much
larger than $1/\texttt{base}$. With `factor = 0.25` and `base = 1e6`, that frequency
is about $10^{-1.5}$. The rotated part resolves distances up to a few hundred tokens,
while the unrotated part is position-free.

The numerical difference is large on real models. For a head of width 512 with
`factor = 0.25` and `base = 1e6` (Gemma 4), the proportional form's slowest rotated
pair turns 0.033 radians per position. The default form with the same three numbers
would rotate 128 channels and give its slowest pair 0.0000012 radians per position.
When `factor = 1`, both functions reduce to the standard half-split RoPE of the whole
head. (In model configs, `base` is called `rope_theta`.)

Both variants preserve the core property of RoPE. Because each rotated pair turns by an
angle proportional to position for both queries and keys, the dot product of a rotated
query at position $p$ and a rotated key at position $q$ depends only on $p - q$. The
unrotated channels add a position-independent term.

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

Padding `freqs` with zeros makes the unrotated pairs fall out of the same rotation
formula, since angle 0 gives $\cos 0 = 1$ and $\sin 0 = 0$. Hugging Face Transformers
uses this same trick, which is why the proportional form builds cosine and sine tables
as wide as the full head while the default form builds tables of width $r$.

## Common mistakes

- **Swapping the frequency denominators.** Using $\texttt{base}^{-2i/D}$ in
  `partial_rope`, or $\texttt{base}^{-2i/(2n)}$ in `proportional_rope`, gives pair 0
  frequency 1 either way. The first channel looks right, and the error grows across the
  rest of the head.
- **Swapping the channel pairing.** Pairing $i$ with $i + n$ inside the first $2n$
  channels of `proportional_rope` rotates channels $n$ through $2n - 1$ instead of
  $D/2$ through $D/2 + n - 1$.
- **Rotating the last $r$ channels.** Some architectures place the rotated slice at the
  end of the head (DeepSeek-V4 does, as covered in `shared-kv-rotation`). The default
  convention here rotates the first $r$ channels.
- **Using adjacent pairs.** Pairing $(0, 1), (2, 3), \dots$ gives the interleaved
  layout of the `rope` problem, which is a different permutation of `x`.
- **Ignoring `factor`.** Rotating all $D$ channels still produces attention scores that
  depend only on relative position, so simple property tests will not catch it.
- **Rounding instead of flooring the pair count.**
  $\texttt{int}(\texttt{factor} \cdot D \mathbin{//} 2)$ floors the result. For
  $D = 12$ and `factor = 0.25`, that gives 1 pair, not 2. Because both counts are float
  expressions in shipped code, a value like `factor = 0.58` with $D = 200$ yields
  `int(115.99999999999999) = 115`, an odd width. This problem restricts `factor` to
  multiples of $1/8$ so the products are exact, though a real model loader should
  validate the resulting count.
- **Ignoring `positions` or rotating clockwise.** Both bugs pass relative-position
  checks on a prompt starting at 0, just as in the `rope` problem.

## Complexity

Both functions run in $O(B\,H\,T\,D)$ time and memory. Only the `(T, r/2)` or
`(T, D/2)` angle table requires `cos` and `sin`, and inference engines typically
precompute those tables per position.

## In the wild

Checked on 2026-11-21 against `transformers` 5.23.0 and each checkpoint's `config.json`
on the Hugging Face Hub.

- **Default with a factor.** `Qwen/Qwen3.5-0.8B` sets `rope_type: "default"`,
  `partial_rotary_factor: 0.25`, `rope_theta: 1e7`, and head width 256.
  `Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters` computes
  `dim = int(head_dim * partial_rotary_factor)` and
  `1 / base ** (arange(0, dim, 2) / dim)`. Then `apply_rotary_pos_emb` in
  `modeling_qwen3_5.py` rotates `q[..., :rotary_dim]` with `rotate_half` and
  concatenates the rest. That matches `partial_rope` with 64 rotated channels and the
  frequency ladder of a 64-wide head. (For images, the model assigns three position IDs
  per token. For text, all three are equal and the code reduces to this.)
- **Proportional.** `google/gemma-4-E2B-it` configures its full-attention layers with
  `rope_type: "proportional"`, `partial_rotary_factor: 0.25`, `rope_theta: 1e6`, and
  head width 512. `_compute_proportional_rope_parameters` in `modeling_rope_utils.py`
  computes `rope_angles = int(rope_proportion * head_dim // 2)`, builds frequencies
  `1 / base ** (arange(0, 2 * rope_angles, 2) / head_dim)`, and pads with zeros to
  `head_dim // 2`. Then `apply_rotary_pos_emb` in `modeling_gemma4.py` applies
  `rotate_half` across the full head, matching `proportional_rope`. Its docstring notes
  that unlike other RoPE helpers, it "will always return an encoding that is the size
  of `head_dim`". Gemma 4's sliding-window layers use plain RoPE with `rope_theta: 1e4`
  and head width 256, so one model uses two position encodings.

The public test "Qwen3.5 and Gemma 4" compares your functions against outputs from
those two code paths (`fixtures/make.py` regenerates them).

A config key can also be present and ignored. The `config.json` of
`LiquidAI/LFM3-1.7B` sets `rope_type: "default"` and `partial_rotary_factor: 0.5` on
its full-attention layers, while the config class in `transformers` 5.23.0 defaults to
`rope_type: "proportional"` with the same factor. Meanwhile, the default path of
`Lfm3RotaryEmbedding` does not read the factor at all and builds 64 nonzero frequencies
for a 128-wide head, producing a full rotation. Because the config alone cannot tell
you which convention the weights were trained with, always compare your frequency table
and a rotated vector against the reference implementation first when porting a model.
