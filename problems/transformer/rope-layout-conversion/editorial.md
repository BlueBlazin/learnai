## The idea

RoPE rotates $D/2$ independent 2D planes inside each attention head by angles proportional to token position. Which two coordinates form plane $i$ is a convention, and two conventions are widely used:

- **Interleaved.** Plane $i$ uses adjacent coordinates $(2i, 2i + 1)$. This matches the RoFormer paper and Meta's Llama reference code, which reshapes the last axis to `(D/2, 2)`, views each pair as a complex number, and multiplies by $e^{\mathrm{i} p \theta_i}$.
- **Half-split.** Plane $i$ uses split coordinates $(i, i + D/2)$. Hugging Face Transformers computes this as `x * cos + rotate_half(x) * sin`, where `rotate_half(x)` splits `x` into halves `x1, x2` and returns `cat(-x2, x1)`, using the doubled angle table `cat(freqs, freqs)`. It requires only two slices and no reshape, which made it popular in PyTorch codebases.

Let $P$ be the permutation that places a vector's even coordinates first and its odd coordinates second, so $(Pz)_i = z_{2i}$ and $(Pz)_{D/2 + i} = z_{2i+1}$. Directly from the definitions,

$$
\operatorname{rope\_half}(Pz,\, p) = P\, \operatorname{rope\_interleaved}(z,\, p) .
$$

The two layouts represent the same linear map in two different coordinate orders. Because a permutation preserves dot products, applying $P$ to both queries and keys before half-split RoPE leaves every attention score unchanged:

$$
\langle \operatorname{rope\_half}(Pq, m),\, \operatorname{rope\_half}(Pk, n) \rangle
= \langle P \operatorname{rope\_interleaved}(q, m),\, P \operatorname{rope\_interleaved}(k, n) \rangle
= \langle \operatorname{rope\_interleaved}(q, m),\, \operatorname{rope\_interleaved}(k, n) \rangle .
$$

Since $Pq = P(W_q x) = (P W_q)\,x$, permuting the output features of a projection is equivalent to permuting the **rows** of its weight matrix. `permute_qk` applies $P$ to the rows of each head in $W_q$ and $W_k$. Nothing else changes. Values are never rotated, so $W_v$ and $W_o$ stay untouched, and attention scores are scalar dot products that do not depend on coordinate order.

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

Reshaping `w` exposes the three indices packed into each row number (head, pair, and member). Transposing axes 1 and 2 swaps the order of pair and member within each head, while keeping heads isolated on axis 0 so rows never mix across heads.

## Common mistakes

- **Using the wrong layout for a checkpoint.** Running `rope_half` on weights trained for `rope_interleaved` (or vice versa) computes a different function, as shown in the worked example, while still returning the right shape and preserving vector norms.
- **Applying the inverse permutation.** Reshaping to `(n_heads, 2, d_head // 2, d_in)` and transposing converts half-split to interleaved instead. For `d_head = 4`, the permutation `0, 2, 1, 3` is its own inverse, so both directions agree. The bug only appears for `d_head >= 6`, so always test with a larger head width.
- **Permuting across heads.** Treating `w` as one head of width `n_heads * d_head` moves the even rows of every head to the front, mixing features across head boundaries.
- **Permuting the input axis.** The columns of $W_q$ index the residual stream rather than head coordinates. This mistake causes a shape error on rectangular weights, but it passes silently on square query projections.
- **Mixing the angle tables.** Interleaved RoPE repeats each frequency for adjacent features (`repeat_interleave`), whereas half-split RoPE repeats the full frequency vector (`cat(freqs, freqs)`). Using `repeat_interleave` with `rotate_half` rotates feature $j$ using the frequency of pair $\lfloor j/2 \rfloor$, giving the two members of a plane different angles so the map is no longer a rotation.
- **Putting the minus sign on the wrong half of `rotate_half`.** Computing `cat(x2, -x1)` rotates clockwise instead of counterclockwise. Scores between half-split vectors still depend only on relative position, so a relative-position check will not catch it, but the scores correspond to $m - n$ instead of $n - m$ and disagree with the interleaved model.

One final caller-side bug is converting only `wq` or only `wk`. Doing so pairs coordinate $2i$ of the query with coordinate $i$ of the key, which breaks the attention-score equivalence checked in the tests.

## Complexity

Each rotation runs in $O(B\,H\,T\,D)$ time and memory. `permute_qk` reindexes `n_heads * d_head` rows in $O(\texttt{n\_heads} \cdot \texttt{d\_head} \cdot \texttt{d\_in})$ time once during checkpoint conversion, with zero overhead at inference time.

## In the wild

Meta's reference Llama implementation (`llama/model.py`, in `precompute_freqs_cis` and `apply_rotary_emb`) uses interleaved RoPE. It builds unit complex numbers with `torch.polar` at frequencies `theta ** (-arange(0, dim, 2) / dim)` and multiplies them against queries and keys reshaped to `(..., D/2, 2)` via `view_as_complex`.

Hugging Face Transformers' Llama (`modeling_llama.py`, in `rotate_half` and `apply_rotary_pos_emb` with `emb = torch.cat((freqs, freqs), dim=-1)`) uses half-split RoPE. Its conversion script `convert_llama_weights_to_hf.py` defines

```python
def permute(w, n_heads, dim1=dim, dim2=dim):
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

and applies it only to `wq` (with `n_heads`) and `wk` (with the number of key/value heads). Converters that export from Hugging Face format back to an interleaved engine apply the inverse permutation. MLX supports both layouts through a single flag, with `mlx.nn.RoPE(dims, traditional=True)` using interleaved layout and the default using half-split (verified against this problem's references on MLX 0.32.4). Granite 4.0 was trained directly in Transformers with half-split RoPE, which `real-model-block` verifies against the checkpoint.
