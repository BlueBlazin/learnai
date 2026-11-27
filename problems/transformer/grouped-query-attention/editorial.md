## The idea

Standard multi-head attention gives every head its own queries, keys, and values.
Grouped-query attention keeps all $H$ query heads while projecting keys and values into
only $H_{kv}$ heads. Each key/value head serves a group of $G = H / H_{kv}$ consecutive
query heads, so query head $h$ reads key/value head $\lfloor h / G \rfloor$.

GQA is equivalent to standard multi-head attention after repeating each key and value
head $G$ times in place:

$$
\tilde{k}_{b,h,s} = k_{b,\lfloor h/G \rfloor,s}, \qquad
\tilde{v}_{b,h,s} = v_{b,\lfloor h/G \rfloor,s}, \qquad
y = \operatorname{softmax}\!\Big(\frac{q\,\tilde{k}^\top}{\sqrt{d}} + M\Big)\,\tilde{v} ,
$$

where $M$ is the additive mask that adds $0$ at allowed positions and $-\infty$ at
masked positions.

Two properties follow directly from this equivalence. First, the score scale stays
$\sqrt{d}$ because sharing a head does not change its width $d$. Second, the output
still has $H$ distinct heads, because each query head in a group has its own query
vectors and therefore its own attention pattern.

## Reference solution (PyTorch)

```python
def gqa(q, k, v, mask):
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

JAX and MLX spell the repetition `repeat(k, groups, axis=1)` and the masking
`where(mask, scores, -inf)`. When `groups == 1`, the repetition is a no-op and the
function reduces to standard multi-head attention. When $H_{kv} = 1$, it repeats the
single key/value head $H$ times. Neither boundary case needs a branch.

You can also avoid copying `k` and `v` by reshaping the queries into groups instead:

```python
b, h, t, d = q.shape
h_kv = k.shape[1]
q = q.reshape(b, h_kv, h // h_kv, t, d)  # (B, H_kv, G, T, d)
scores = q @ k[:, :, None].transpose(-2, -1) / math.sqrt(d)  # (B, H_kv, G, T, S)
scores = scores.masked_fill(~mask, float("-inf"))
return (torch.softmax(scores, dim=-1) @ v[:, :, None]).reshape(b, h, t, d)
```

Reshaping `(B, H, ...)` to `(B, H_kv, G, ...)` places query head $h$ at index
`[h // G, h % G]`. That matches the exact same grouping while broadcasting `k` and `v`
without copying them.

## Common mistakes

- **`tile` instead of `repeat_interleave`.** Calling `k.tile(1, G, 1, 1)` or
  `k.repeat(1, G, 1, 1)` in PyTorch, or `jnp.tile` in JAX, produces the interleaved
  head order `0, 1, 0, 1` so query head $h$ reads key/value head $h \bmod H_{kv}$. The
  output shape is identical, and the result is even correct when $H_{kv} = 1$ or
  $H_{kv} = H$, so tests with those settings miss it. The equivalent bug in the
  grouped-query formulation is reshaping `q` to `(B, G, H_kv, T, d)` instead of
  `(B, H_kv, G, T, d)`.
- **Dividing by the wrong count.** Query head $h$ maps to key/value head
  $\lfloor h / G \rfloor$ with $G = H / H_{kv}$, not $\lfloor h / H_{kv} \rfloor$. The
  two formulas coincide when $H = H_{kv}^2$ (such as 4 query heads on 2 key/value
  heads), hiding the bug in small tests. When $H_{kv} = 1$, the wrong formula indexes
  past the end of the head axis for every head after head 0.
- **Averaging the queries of a group.** GQA shares keys and values, not queries.
  Collapsing a group to one query produces one attention pattern per key/value head,
  whereas GQA computes $H$ distinct patterns. (The GQA paper averages key and value
  *projection weights* once to initialize shared heads from a multi-head checkpoint
  before fine-tuning, not queries at inference time.)
- **Rescaling for the group.** Scores are always divided by $\sqrt{d}$, regardless of
  $G$.
- **Misreading the mask.** Here `mask` is boolean with `True` meaning "may attend".
  Adding it directly to `scores` (as with an additive `0` / `-inf` mask) shifts allowed
  scores by 1 and masks nothing. When every entry is `True`, that constant shift
  cancels in softmax and hides the bug. Inverting the boolean masks out the allowed
  keys instead. Always check the mask convention.

## Complexity

The computation takes $O(B\,H\,T\,S\,d)$ time and $O(B\,H\,T\,S)$ memory for the
attention weights, identical to standard multi-head attention. GQA does not reduce the
cost of the attention matmuls themselves. Using `repeat_interleave` allocates
$O(B\,H\,S\,d)$ extra memory for the expanded keys and values during the call, whereas
the reshaped broadcast form allocates none.

## In the wild

GQA saves parameter count in the K/V projections and, more importantly, cuts KV cache
memory by storing $H_{kv}$ heads per layer instead of $H$. In Llama 3 8B (32 layers, 32
query heads, 8 key/value heads, head width 128), each token occupies
$32 \cdot 2 \cdot 8 \cdot 128 = 65{,}536$ cached values, or 128 KiB in 16-bit floats.
An 8,192-token context therefore takes 1 GiB instead of the 4 GiB a 32-head cache would
require. Mistral 7B also uses 32 query heads on 8 key/value heads, and Qwen3-0.6B uses
16 on 8. Multi-query attention is the $H_{kv} = 1$ limit, which the first Gemma 2B
used.

Hugging Face Transformers implements the expansion in a helper named `repeat_kv`, which
expands a new axis after the head axis and reshapes, matching
`torch.repeat_interleave(x, dim=1, repeats=n_rep)`. The KV cache always stores the
unexpanded `(B, H_kv, S, d)` tensors because expanding before caching would lose the
entire memory saving. Serving engines that avoid the copy use the reshaped broadcast
form above or a fused attention kernel that indexes the shared head directly.
