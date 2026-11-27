## The idea

Let $R(p)$ denote the rotation of the trailing `rope_dim` channels by position $p$.
Because $R(p)$ is a linear orthogonal transformation, it preserves dot products and
satisfies $R(a)\,R(b) = R(a + b)$. Three consequences follow directly.

**Scores depend only on relative position**, just as in standard RoPE. For query position
$p_t = \texttt{pos\_q}[t]$ and key position $p_s = \texttt{pos\_k}[s]$:

$$
\langle R(p_t)\,q,\ R(p_s)\,k \rangle = \langle q,\ R(p_s - p_t)\,k \rangle .
$$

**The raw weighted sum does not.** With separate keys and values, the values are never
rotated. Here the value *is* the rotated key, so the weighted sum mixes vectors that
each carry an absolute rotation:

$$
o_t = \sum_s a_{t,s}\, R(p_s)\,k_s .
$$

If you shift the entire context forward by 100 positions, the attention weights stay the
same, but every $R(p_s)$ changes and alters $o_t$.

**One inverse rotation restores relative positioning.** By linearity of $R$:

$$
R(-p_t)\, o_t = \sum_s a_{t,s}\, R(p_s - p_t)\,k_s .
$$

Every cached row now reaches the query rotated by its relative offset $p_s - p_t$, so
the output depends only on relative positions. That angle also carries useful
information. In the rotary channels, the model receives each value encoded with how far
away it was written, which unrotated values do not provide.

Because the KV cache stores the rotated rows (used as both keys and values), the
inverse rotation must happen once per query on the attention output rather than on the
cache itself.

Note that this problem calls the rotary frequency base `base` (matching `rope`), while
model configurations call it `rope_theta`.

## Reference solution (PyTorch)

```python
def _rotate(x, positions, rope_dim, base):
    freqs = base ** (-torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    angles = positions.to(torch.float32)[:, None] * freqs
    cos, sin = torch.cos(angles), torch.sin(angles)
    split = x.shape[-1] - rope_dim
    plain, tail = x[..., :split], x[..., split:]
    a, b = tail[..., 0::2], tail[..., 1::2]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).reshape(tail.shape)
    return torch.cat((plain, rotated), dim=-1)


def shared_kv_attention(q, kv, pos_q, pos_k, mask, rope_dim, base):
    q = _rotate(q, pos_q, rope_dim, base)
    kv = _rotate(kv, pos_k, rope_dim, base)
    scores = q @ kv.T / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ kv
    return _rotate(out, -pos_q, rope_dim, base)
```

Using `x.shape[-1] - rope_dim` instead of `-rope_dim` simply keeps the two slices
visibly complementary. Both forms are equivalent for `rope_dim >= 2`.

## Common mistakes

- **Omitting the inverse rotation.** The code looks like ordinary attention and
  produces plausible numbers, but its output shifts whenever the same tokens appear
  later in the sequence.
- **Using the wrong sign on the output rotation.** Rotating by $+p_t$ instead of $-p_t$
  doubles the absolute position rotation instead of canceling it.
- **Using unrotated `kv` as the values (or de-rotating each row by its own position).**
  The output becomes shift-invariant, so a shift-invariance check passes, yet the
  values are wrong because every row arrives with relative angle 0 instead of
  $p_s - p_t$. The single-key case catches this directly.
- **Using unrotated values while still applying the inverse rotation.** Every value
  arrives rotated by $-p_t$, reintroducing an absolute position dependence.
- **Rotating the wrong slice.** Rotating the leading `rope_dim` channels fails here.
  Architectures differ on slice placement (Qwen3.5 puts the rotary slice first, as in
  `partial-rope`), whereas this layer puts it last.
- **Wrong pair layout or frequency ladder.** Splitting the rotary slice in half instead
  of taking adjacent interleaved pairs, or dividing the frequency exponent by $d$
  instead of `rope_dim`, gives wrong angles.
- **Using the wrong positions for the inverse rotation.** Using `arange(T)` works when
  a prompt starts at 0 but fails on every decoding step. Using `pos_k` works only when
  queries and keys are the exact same tokens, and breaks as soon as a query attends
  over a longer cache.
- **Wrong scale or mask.** The scale factor is $d^{-1/2}$ using the full head width
  $d$. Masked entries must be excluded from the softmax using $-\infty$ (or by zeroing
  their weights). Setting masked scores to 0 or subtracting a fixed constant such as 30
  leaves them in the softmax, so a masked row with a large score still leaks through.

## Complexity

Computing the attention scores and weighted sum takes $O(H\,T\,S\,d)$ time. The three
rotations take $O((H\,T + S)\,\texttt{rope\_dim})$ time. The shared cache stores
$S \cdot d$ numbers, compared with $2\,H_{kv}\,S\,d$ for separate keys and values
across $H_{kv}$ KV heads.

## In the wild

Verified on 2026-11-21 in `transformers` 5.23.0
(`models/deepseek_v4/modeling_deepseek_v4.py`) and in the `config.json` of
`deepseek-ai/DeepSeek-V4-Pro-0813` (`num_key_value_heads: 1`, `head_dim: 512`,
`qk_rope_head_dim: 64`, `rope_theta: 10000`, 128 attention heads).

- `DeepseekV4Attention.forward` projects to a single key-value head, rotates `q` and
  `kv` with `apply_rotary_pos_emb`, passes `kv` as both key and value, and then calls
  `apply_rotary_pos_emb(attn_output, cos, -sin)`. Negating the sine table rotates by
  minus the query position. The inline comment notes: "K=V in V4, so V picked up rope
  on its trailing rope slice."
- `apply_rotary_pos_emb` rotates `x[..., -rope_dim:]` using a `rotate_half` helper that
  pairs `x[..., 0::2]` with `x[..., 1::2]`, giving a trailing slice with interleaved
  pairs. (The class docstring mentions "first `rope_head_dim`", but the code and helper
  docstring use the trailing slice.)
- Frequencies come from `DeepseekV4RotaryEmbedding.compute_default_rope_parameters`
  with `dim = int(head_dim * partial_rotary_factor)`, matching the slice width.
- The attention scale is `head_dim ** -0.5`.

Several pieces of `DeepseekV4Attention` are omitted here to keep the problem focused.
Those include the low-rank query projection and norms that produce `q` and `kv`, a
learned per-head attention sink that participates in the softmax and is then dropped,
compressed cache entries rotated with a rescaled base (as in `yarn-frequencies`), and
the grouped output projection. The problem fixture runs the library's rotation and
attention functions at model dimensions with sinks disabled (`fixtures/make.py`). Every
detail above comes directly from the code rather than the paper.
