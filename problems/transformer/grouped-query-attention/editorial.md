## The idea

Standard multi-head attention gives every head its own queries, keys, and values.
Grouped-query attention keeps all $H$ query heads and projects the keys and values
into only $H_{kv}$ heads; each key/value head serves a group of $G = H / H_{kv}$
consecutive query heads. Query head $h$ reads key/value head $\lfloor h / G \rfloor$.

So GQA is exactly standard attention applied to keys and values in which every head
has been repeated $G$ times in place:

$$
\tilde{k}_{b,h,s} = k_{b,\lfloor h/G \rfloor,s}, \qquad
\tilde{v}_{b,h,s} = v_{b,\lfloor h/G \rfloor,s}, \qquad
y = \operatorname{softmax}\!\Big(\frac{q\,\tilde{k}^\top}{\sqrt{d}} + M\Big)\,\tilde{v} .
$$

That equivalence is the specification, and one of the tests states it directly. Two
things follow from it. The scale stays $\sqrt{d}$, the width of one head: sharing a
head does not make it wider. And the output has $H$ heads, each with its own attention
pattern, because the queries differ within a group even though the keys do not.

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
`where(mask, scores, -inf)`. With `groups == 1` the repetition does nothing and the
function is standard attention; with one key/value head it repeats that head $H$
times. Neither end needs a branch.

The repetition can be avoided altogether by grouping the queries instead:

```python
b, h, t, d = q.shape
h_kv = k.shape[1]
q = q.reshape(b, h_kv, h // h_kv, t, d)  # (B, H_kv, G, T, d)
scores = q @ k[:, :, None].transpose(-2, -1) / math.sqrt(d)  # (B, H_kv, G, T, S)
scores = scores.masked_fill(~mask, float("-inf"))
return (torch.softmax(scores, dim=-1) @ v[:, :, None]).reshape(b, h, t, d)
```

Reshaping `(B, H, ...)` to `(B, H_kv, G, ...)` puts query head $h$ at
`[h // G, h % G]`, which is the same grouping, and the keys and values are only
broadcast, never copied.

## Common mistakes

- **`tile` instead of `repeat_interleave`.** `k.tile(1, G, 1, 1)`, `k.repeat(1, G, 1,
  1)` in PyTorch, and `jnp.tile` produce the head order `0, 1, 0, 1`: query head $h$
  then reads key/value head $h \bmod H_{kv}$. The shape is identical, and the result
  is even correct for $H_{kv} = 1$ and for $H_{kv} = H$, so a test with one of those
  two settings does not notice. The mirror image of this mistake is reshaping the
  queries to `(B, G, H_kv, T, d)` instead of `(B, H_kv, G, T, d)`.
- **Dividing by the wrong count.** The key/value head of query head $h$ is
  $\lfloor h / G \rfloor$ with $G = H / H_{kv}$, not $\lfloor h / H_{kv} \rfloor$.
  The two coincide when $H = H_{kv}^2$, such as the tempting toy setting of 4 heads on
  2. With a single key/value head the wrong formula sends every query head except the
  first past the end of the axis.
- **Averaging the queries of a group.** GQA shares keys and values, not queries.
  Collapsing a group to one query computes one attention pattern per key/value head;
  the model has $H$ distinct ones. (The GQA paper does average, but it averages the
  key and value *projection weights* of a multi-head checkpoint to initialize the
  shared heads, once, before fine-tuning. Nothing is averaged at inference time.)
- **Rescaling for the group.** The scores are divided by $\sqrt{d}$ for any $G$.
- **Misreading the mask.** Here it is boolean with `True` meaning "may attend".
  Adding it to the scores (as one does with an additive mask of `0` and `-inf`) shifts
  the allowed scores by 1 and hides nothing; with a mask that is all `True` the shift
  cancels in the softmax and the bug is invisible. Using it with the opposite meaning
  hides the allowed keys. Always check which convention a mask argument follows.

## Complexity

$O(B\,H\,T\,S\,d)$ time and $O(B\,H\,T\,S)$ memory for the attention weights, the same
as standard attention: GQA does not make the attention itself cheaper. With
`repeat_interleave` the expanded keys and values cost $O(B\,H\,S\,d)$ extra memory for
the duration of the call; the grouped formulation needs none.

## In the wild

What GQA saves is the projection and, above all, the KV cache, which stores $H_{kv}$
heads per layer instead of $H$. For Llama 3 8B (32 layers, 32 query heads, 8 key/value
heads, head width 128) one token occupies $32 \cdot 2 \cdot 8 \cdot 128 = 65{,}536$
cached numbers, 128 KiB in 16-bit floats, so a context of 8,192 tokens costs 1 GiB.
With 32 key/value heads it would cost 4 GiB. Mistral 7B uses the same 32 on 8, and
Qwen3-0.6B uses 16 on 8. Multi-query attention is the limit of one key/value head; the
first Gemma 2B used it.

Hugging Face Transformers implements the expansion in a helper called `repeat_kv`,
which expands a new axis after the head axis and reshapes, and documents itself as the
equivalent of `torch.repeat_interleave(x, dim=1, repeats=n_rep)`. The cache always
stores the unexpanded `(B, H_kv, S, d)` tensors; expanding before storing would throw
away the entire saving. Engines that care about the copy use the broadcast form above
or a fused attention kernel that indexes the shared head directly.
