## The idea

An attention score is a dot product, $q \cdot k = \lVert q \rVert\,\lVert k \rVert \cos\angle(q, k)$, where $q, k \in \mathbb{R}^d$ are a query and key vector of head dimension $d$. Because the dot product multiplies directional alignment by vector length, unbounded lengths can distort attention. A single long key can dominate the softmax for every query, and in deep models growing norms can push the softmax toward one-hot saturation.

QK-norm removes raw vector length before the dot product. Each query and key is divided by its own root mean square (with smoothing constant $\varepsilon$) and multiplied elementwise by a learned per-dimension gain $g^q$ or $g^k$:

$$
\hat{q} = \frac{q}{\operatorname{rms}(q)} \odot g^q, \qquad
\hat{k} = \frac{k}{\operatorname{rms}(k)} \odot g^k, \qquad
\operatorname{rms}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_{i=0}^{d-1} x_i^2 + \varepsilon}.
$$

After normalization, the scaled score $\hat{q} \cdot \hat{k} / \sqrt{d}$ is bounded by $\sqrt{d}\,\max_i |g^q_i|\,\max_i |g^k_i|$ regardless of input magnitude. The learned gains alone govern how sharp the attention distribution can become.

Three design details determine the exact computation:

- **Normalization axis.** Normalize over the head dimension `d`, not the full hidden dimension. Each head runs an independent attention calculation, so every `(batch, head, token)` vector gets its own root mean square. In a model checkpoint, a gain tensor of shape `(d,)` confirms per-head normalization.
- **Target tensors.** Normalize only queries and keys. Values carry the content being aggregated, and their magnitudes encode information the model relies on.
- **Order relative to RoPE.** Normalize before applying RoPE. Because a 2D rotation preserves the root mean square, dividing by $\operatorname{rms}(x)$ commutes with RoPE. However, the learned gain scales dimensions $2i$ and $2i + 1$ by different values, and RoPE mixes those two dimensions. Scaling then rotating produces a different linear map than rotating then scaling whenever the two gains in a pair differ. Applying RoPE last also preserves relative-position dependence in the final dot product.

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

For inputs of shape `(B, H, T, d)` (batch size `B`, `H` heads, sequence length `T`, and head dimension `d`), taking `mean(dim=-1, keepdim=True)` produces shape `(B, H, T, 1)`, giving one scalar per query or key vector. The rotation tables have shape `(T, d/2)` and broadcast automatically against `(B, H, T, d/2)`. The JAX and MLX implementations follow the same structure, using `jnp.where` or `mx.where` for causal masking.

## Common mistakes

- **Normalizing over the full hidden size.** Applying RMSNorm to the full projection of width $H \cdot d$ before splitting into heads (or averaging over both the head and feature axes) produces one root mean square per token. Heads then share a single scale, so one head with large activations shrinks every other head. Hugging Face's Qwen3 code explicitly notes this with the comment `"unlike olmo, only on the head dim!"`.
- **Normalizing after RoPE.** With all-ones gains, normalizing before or after RoPE gives the same result, so naive initialization tests pass. Once the gains are trained and differ across dimensions, the two orders compute different functions.
- **Normalizing the values too.** Attention outputs are weighted averages of `v`, so normalizing `v` rescales the layer output directly.
- **Using L2 norm instead of RMS.** Dividing by $\sqrt{\sum_i x_i^2}$ instead of $\sqrt{\frac{1}{d}\sum_i x_i^2}$ shrinks every vector by $\sqrt{d}$ and every score by $d$. Although earlier work defined QK-norm using L2 normalization with a learned temperature (Henry et al., 2020), modern architectures such as Qwen3 and Gemma 3 use RMSNorm.
- **Dropping the $1/\sqrt{d}$ scale.** Because the vectors are already normalized, the scale factor looks redundant. Removing it still breaks equivalence with the function the gains were trained under.
- **Sharing one gain for both queries and keys.** `q_norm` and `k_norm` are separate layers with distinct weight vectors.
- **Placing epsilon outside the square root.** The formula requires $\sqrt{\text{mean} + \varepsilon}$, not $\sqrt{\text{mean}} + \varepsilon$. For typical vector magnitudes and $\varepsilon = 10^{-6}$, the two agree to roughly six digits, making the bug easy to miss on small tests.

## Complexity

For batch size $B$, $H$ heads, sequence length $T$, and head dimension $d$, both RMSNorm and RoPE run in $O(B\,H\,T\,d)$ time. That cost is small compared to attention itself, which takes $O(B\,H\,T^2\,d)$ time and $O(B\,H\,T^2)$ memory.

## In the wild

In Hugging Face's Qwen3 implementation, each tensor is reshaped into heads before normalization: `q_norm(q_proj(x).view(B, T, H, d)).transpose(1, 2)`, with the same pattern for `k_norm` and no norm on `v`. Both norms are RMSNorm layers of width `head_dim`. `apply_rotary_pos_emb` runs next, followed by standard `head_dim ** -0.5` scaling.

Google's PyTorch reference for Gemma 3 uses the same sequence (reshape into heads, apply `query_norm` and `key_norm`, then apply RoPE). One checkpoint detail to watch for is that Gemma's RMSNorm stores its gain as an offset from 1 and multiplies by `1 + weight`. The Gemma 3 technical report notes that QK-norm replaces the attention logit soft-capping used in Gemma 2.

During autoregressive decoding with a KV cache, keys are cached after normalization and rotation. As a result, QK-norm runs only once per token and layer.
