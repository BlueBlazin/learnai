## The idea

An attention score is a dot product, $q \cdot k = \lVert q \rVert\,\lVert k \rVert \cos\angle(q, k)$,
so it mixes two things: how well the directions match and how long the vectors are.
Nothing bounds the lengths. A key that happens to be long wins the softmax against
every query, and in a deep model the lengths can grow until the softmax saturates
into one-hot attention.

QK-norm takes the lengths out. Each query and each key is divided by its root mean
square, so every normalized vector has the same size, and a learned gain per dimension
puts back a controlled amount of scale:

$$
\hat{q} = \frac{q}{\operatorname{rms}(q)} \odot g^q, \qquad
\hat{k} = \frac{k}{\operatorname{rms}(k)} \odot g^k, \qquad
\operatorname{rms}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_i x_i^2 + \varepsilon} .
$$

The score $\hat{q} \cdot \hat{k} / \sqrt{d}$ is now bounded by
$\sqrt{d}\,\max_i |g^q_i|\,\max_i |g^k_i|$ whatever the inputs: the gains, which the
model learns, decide how sharp the attention can get.

The three questions of the problem have these answers.

**Which axis?** The head dimension. Each head is a separate attention computation, and
its queries and keys are normalized on their own, one root mean square per
`(batch, head, token)`. The gain has shape `(d,)`, the width of a head, which is the
giveaway when you read a checkpoint.

**Applied to what?** Queries and keys. The values carry the content that is mixed, and
their magnitude is information the model uses.

**Before or after RoPE?** Before. A rotation preserves the root mean square, so the
division alone would commute with it. The gain does not: it scales dimension $2i$ and
dimension $2i + 1$ by different numbers, and the rotation mixes those two dimensions.
Scaling then rotating and rotating then scaling are different linear maps unless the
two gains of every pair are equal. Normalizing first also keeps what makes RoPE work:
both vectors are rotated last, so the score still depends on positions only through
their difference.

## Reference solution (PyTorch)

```python
def _rms_norm(x, weight, eps):
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


def _rotate(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.reshape(x.shape)


def qk_norm_attention(q, k, v, q_weight, k_weight, cos, sin, eps=1e-6):
    seq, dim = q.shape[-2], q.shape[-1]
    q = _rotate(_rms_norm(q, q_weight, eps), cos, sin)
    k = _rotate(_rms_norm(k, k_weight, eps), cos, sin)
    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

`mean(dim=-1, keepdim=True)` gives shape `(B, H, T, 1)`, one value per vector. The
tables of shape `(T, d/2)` broadcast against `(B, H, T, d/2)` from the right. The JAX
and MLX versions are direct translations with `jnp.where` and `mx.where` for the mask.

## Common mistakes

- **Normalizing over the full hidden size.** Applying the norm to the projection
  output of width $H \cdot d$ before splitting it into heads, or averaging over the
  head axis as well, computes one root mean square per token. Heads then share a
  scale: one head with large activations shrinks all the others. The comment in the
  Hugging Face Qwen3 code exists for this reason: "unlike olmo, only on the head dim!"
- **Normalizing after RoPE.** With unit gains the result is identical, so a test with
  freshly initialized gains passes. With trained gains it is a different function.
- **Normalizing the values too.** It is called QK-norm for a reason. The output is a
  weighted average of the values, so normalizing them rescales the output directly.
- **L2 norm instead of RMS.** Dividing by $\sqrt{\sum_i x_i^2}$ instead of
  $\sqrt{\tfrac{1}{d}\sum_i x_i^2}$ makes every vector shorter by $\sqrt{d}$ and every
  score smaller by a factor $d$. Some papers do define QK-norm with the L2 norm and a
  learned temperature (Henry et al., 2020); the models named here use RMSNorm layers.
- **Dropping $1/\sqrt{d}$.** The vectors are normalized, so the scale looks redundant.
  It is still part of the function the gains were trained under.
- **One gain for both.** `q_norm` and `k_norm` are separate layers with separate
  gains.
- **Epsilon in the wrong place.** $\sqrt{\text{mean} + \varepsilon}$, not
  $\sqrt{\text{mean}} + \varepsilon$. With $\varepsilon = 10^{-6}$ and vectors of
  ordinary size the two agree to six digits, which is how the wrong one survives.

## Complexity

The normalization and the rotation are $O(B\,H\,T\,d)$, negligible next to the
attention itself, $O(B\,H\,T^2\,d)$ time and $O(B\,H\,T^2)$ memory.

## In the wild

Hugging Face's Qwen3 attention is one line per tensor:
`q_norm(q_proj(x).view(B, T, H, d)).transpose(1, 2)`, the same for the keys with
`k_norm`, and no norm on the values; `apply_rotary_pos_emb` comes after, and the
scaling is `head_dim ** -0.5`. Both norms are RMSNorm layers of size `head_dim`.
Google's PyTorch reference for Gemma 3 has the same order: reshape into heads,
`query_norm` and `key_norm`, then the rotary embedding. Gemma's RMSNorm stores its
gain as an offset from one (the layer multiplies by `1 + weight`), so a loader must
know which convention a gain tensor follows. The Gemma 3 report describes QK-norm as
the replacement for the soft-capping of attention logits used in Gemma 2.

With a KV cache, the keys are stored after normalization and rotation, exactly as they
enter the dot product, so the cost of QK-norm is paid once per token and layer.
