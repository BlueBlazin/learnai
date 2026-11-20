## The idea

Write $R(p)$ for the rotation of the rotary slice by position $p$. It is linear, it
preserves dot products, and $R(a)\,R(b) = R(a + b)$. Three facts follow.

**Scores are relative**, as in any RoPE:

$$
\langle R(p_t)\,q,\ R(p_s)\,k \rangle = \langle q,\ R(p_s - p_t)\,k \rangle .
$$

**Values are not.** With separate keys and values, the value is never rotated and
none of this arises. Here the value *is* the rotated key, so the weighted sum is

$$
o_t = \sum_s a_{t,s}\, R(p_s)\,k_s ,
$$

a mix of rows that each carry an absolute rotation. Move the whole context 100
positions along and every $R(p_s)$ changes, although the weights do not.

**One more rotation fixes it.** Because $R$ is linear,

$$
R(-p_t)\, o_t = \sum_s a_{t,s}\, R(p_s - p_t)\,k_s .
$$

Now every row reaches the query turned by its *distance* to the query, and the
output is a function of relative positions only. This is more than a cleanup: the
angle $p_s - p_t$ carries information. In the rotary channels the model receives each
value together with how far away it was written, which separate keys and values do
not provide.

The cache stores the rotated rows (they are both keys and values), so the inverse
rotation has to happen on the output, once per query, not on the cache.

`base` is the argument name of the `rope` problem; configurations call it
`rope_theta`.

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

`x.shape[-1] - rope_dim` is used for the split instead of `x[..., -rope_dim:]` only to
keep the two slices visibly complementary; both spellings are right for
`rope_dim >= 2`.

## Common mistakes

- **No inverse rotation.** The most natural bug: the function looks like ordinary
  attention and returns plausible numbers. They change when the same tokens appear
  later in the context.
- **The wrong sign.** Rotating the output by $+p_t$ doubles the absolute rotation.
- **De-rotating each value by its own position** (equivalently, using the unrotated
  tensor as the value). The output is then shift-invariant, so a test of that
  property passes, and it is still wrong: every row arrives with relative angle 0
  instead of $p_s - p_t$. The single-key case tests this directly.
- **Unrotated values with the inverse rotation still applied.** Every row arrives
  turned by $-p_t$, an absolute angle again.
- **The wrong slice.** Rotating the first `rope_dim` channels. Models disagree on
  where the rotary slice sits (Qwen3.5 puts it first, see `partial-rope`); this one
  puts it last.
- **The wrong pairing or ladder.** Half-split pairs inside the slice, or the exponent
  divided by $d$ instead of `rope_dim`.
- **The wrong positions on the way out.** `arange(T)` is right for a prompt that
  starts at 0 and wrong for every decoding step. The positions of the cached rows
  are right in self-attention over a whole prompt, where queries and rows are the
  same tokens, and wrong as soon as one new query attends over a longer cache.
- **Scale and mask.** The scale is $d^{-1/2}$ with $d$ the whole head width. A
  masked row is excluded from the softmax: scoring it 0, or subtracting a fixed
  amount such as 30 from its score, leaves it in the sum, and a hidden row whose
  score is high enough still wins. Use $-\infty$ (or a `where` on the weights).

## Complexity

$O(H\,T\,S\,d)$ time for the scores and the weighted sum; the three rotations are
$O((H\,T + S)\,\texttt{rope\_dim})$. The cache holds $S \cdot d$ numbers, where
separate keys and values for $H_{kv}$ heads would hold $2\,H_{kv}\,S\,d$.

## In the wild

Checked on 2026-11-21 in `transformers` 5.23.0,
`models/deepseek_v4/modeling_deepseek_v4.py`, and in the `config.json` of
`deepseek-ai/DeepSeek-V4-Pro-0813` (`num_key_value_heads: 1`, `head_dim: 512`,
`qk_rope_head_dim: 64`, `rope_theta: 10000`, 128 attention heads).

- `DeepseekV4Attention.forward` projects to a single key-value head, rotates `q` and
  `kv` with `apply_rotary_pos_emb`, passes `kv` as both key and value, and then calls
  `apply_rotary_pos_emb(attn_output, cos, -sin)`: the same tables with the sine
  negated, which is the rotation by minus the query position. Its comment reads
  "K=V in V4, so V picked up rope on its trailing rope slice."
- `apply_rotary_pos_emb` there rotates `x[..., -rope_dim:]` with a `rotate_half` that
  pairs `x[..., 0::2]` with `x[..., 1::2]`: trailing slice, interleaved pairs. (The
  class docstring says "first `rope_head_dim`"; the code and its helper's docstring
  say trailing, and the code is what runs.)
- The frequencies come from `DeepseekV4RotaryEmbedding.compute_default_rope_parameters`
  with `dim = int(head_dim * partial_rotary_factor)`, the ladder of the slice width.
- The scale is `head_dim ** -0.5`.

Left out of this problem, all present in that module: the low-rank query projection
and the norms that produce `q` and `kv`, a learned per-head attention sink that takes
part in the softmax and is then dropped, the compressed entries that the model
appends to the rows (their rotation uses a different base, rescaled as in
`yarn-frequencies`), and the grouped output projection. The fixture of this problem
runs the library's rotation and attention functions on the model's sizes with the
sinks disabled (`fixtures/make.py`). The model's paper was not consulted; everything
here is from the code.
