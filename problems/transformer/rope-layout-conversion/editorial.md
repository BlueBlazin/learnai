## The idea

RoPE turns each of $D/2$ two-dimensional planes of a head by an angle proportional
to the position. Which two coordinates span plane $i$ is a bookkeeping choice, and
two choices are in circulation:

- **interleaved**: plane $i$ is coordinates $(2i, 2i + 1)$. This is how the RoFormer
  paper writes the rotation matrix, and how Meta's Llama code implements it: it
  reshapes the last axis to `(D/2, 2)`, views each pair as a complex number, and
  multiplies by $e^{\mathrm{i} p \theta_i}$.
- **half-split**: plane $i$ is coordinates $(i, i + D/2)$. This is what Hugging Face
  Transformers computes with `x * cos + rotate_half(x) * sin`, where
  `rotate_half(x) = cat(-x2, x1)` over the two halves and the angle table is
  `cat(freqs, freqs)`. It needs no reshape, only two slices, which is why it became
  the common form.

Let $P$ be the permutation that lists a vector's even coordinates first and its odd
coordinates second, $(Pz)_i = z_{2i}$ and $(Pz)_{D/2 + i} = z_{2i+1}$. Then, directly
from the definitions,

$$
\operatorname{rope\_half}(Pz,\, p) = P\, \operatorname{rope\_interleaved}(z,\, p) .
$$

The two layouts are the same linear map in two coordinate systems. A permutation
preserves dot products, so if the queries and keys of a model are both passed
through $P$ before the half-split rotation, every score is what it was:

$$
\langle \operatorname{rope\_half}(Pq, m),\, \operatorname{rope\_half}(Pk, n) \rangle
= \langle P \operatorname{rope\_interleaved}(q, m),\, P \operatorname{rope\_interleaved}(k, n) \rangle
= \langle \operatorname{rope\_interleaved}(q, m),\, \operatorname{rope\_interleaved}(k, n) \rangle .
$$

And $Pq = P(W_q x)$ is $(P W_q)\,x$: permuting the coordinates of the projection's
output is permuting the **rows** of its weight. That is `permute_qk`: apply $P$ to
the rows of each head of $W_q$ and of $W_k$. Nothing else changes. The values are
not rotated, so $W_v$ and $W_o$ stay as they are, and the scores, being dot
products, do not care in which order the head's coordinates are stored.

## Reference solution (PyTorch)

```python
def _cos_sin(dim, pos, theta):
    freqs = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = pos.to(torch.float32)[:, None] * freqs  # (T, D/2)
    return torch.cos(angles), torch.sin(angles)


def rope_interleaved(x, pos, theta=10000.0):
    cos, sin = _cos_sin(x.shape[-1], pos, theta)
    a, b = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return rotated.reshape(x.shape)


def rope_half(x, pos, theta=10000.0):
    cos, sin = _cos_sin(x.shape[-1], pos, theta)
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)


def permute_qk(w, d_head, n_heads):
    pairs = w.reshape(n_heads, d_head // 2, 2, w.shape[-1])  # [head, pair, member, in]
    return pairs.transpose(1, 2).reshape(w.shape)  # [head, member, pair, in]
```

The reshape names the three indices hidden in a row number, and the transpose
changes which of the last two varies fastest. Rows of different heads are in
different slices of the first axis and cannot mix.

## Common mistakes

- **The wrong layout for the checkpoint.** `rope_half` on weights trained for
  `rope_interleaved` (or the reverse) is a different function of the same input, as
  the worked example shows. Both return the right shape and preserve norms.
- **The inverse permutation.** Reshaping to `(n_heads, 2, d_head // 2, d_in)` and
  transposing converts half-split to interleaved. For `d_head = 4` the permutation
  `0, 2, 1, 3` is its own inverse and both directions give the same matrix; from
  `d_head = 6` on they differ. Test with a realistic width.
- **Permuting across heads.** Treating the matrix as one head of width
  `n_heads * d_head` moves the even rows of all heads to the front, so rows cross
  head boundaries and most heads receive pairs from another head.
- **Permuting the input axis.** The columns of $W_q$ index the residual stream, not
  the head's coordinates. The mistake fails on shapes unless the weight is square,
  which query projections usually are.
- **Mixing the angle tables.** The interleaved table repeats each angle for two
  adjacent features (`repeat_interleave`), the half-split table repeats the whole
  list (`cat(freqs, freqs)`). Using the first with `rotate_half` turns feature $j$
  at the frequency of pair $\lfloor j/2 \rfloor$, so the two members of a plane get
  different angles and the result is not a rotation at all.
- **`rotate_half` with the sign on the wrong half.** `cat(x2, -x1)` rotates
  clockwise. Scores among half-split vectors still depend on relative position
  only, so the bug hides from that test; they are the scores for $m - n$ instead of
  $n - m$, and do not match the interleaved model.

Converting only one of the two matrices is the remaining classic: scores then pair
coordinate $2i$ of the query with coordinate $i$ of the key. This function converts
one matrix at a time, so that mistake lives in the caller; the score property in the
tests is what it would fail.

## Complexity

Each rotation is $O(B\,H\,T\,D)$ time and memory. The permutation is a reindexing of
`n_heads * d_head` rows, done once when a checkpoint is converted, never at
inference time.

## In the wild

Meta's reference implementation (`llama/model.py`: `precompute_freqs_cis`,
`apply_rotary_emb`) builds `torch.polar` unit complex numbers at the frequencies
`theta ** (-arange(0, dim, 2) / dim)` and multiplies them with
`view_as_complex` of the queries and keys reshaped to `(..., D/2, 2)`: interleaved.
Transformers' Llama (`modeling_llama.py`: `rotate_half`, `apply_rotary_pos_emb`,
`emb = torch.cat((freqs, freqs), dim=-1)`) is half-split, and its
`convert_llama_weights_to_hf.py` contains

```python
def permute(w, n_heads, dim1=dim, dim2=dim):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

applied to `wq` with the number of heads and to `wk` with the number of key/value
heads, and to nothing else. Converters that go from the Hub format back to an
interleaved engine apply the inverse. MLX has both layouts behind one flag:
`mlx.nn.RoPE(dims, traditional=True)` rotates adjacent pairs, and the default rotates
feature $i$ with $i + D/2$ (checked against this problem's references on MLX
0.32.4). Granite 4.0 was trained in Transformers and is half-split natively, which is
what `real-model-block` verifies against the checkpoint.
