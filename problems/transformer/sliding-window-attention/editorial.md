## The idea

A causal mask says "not the future": $j \le i$. A sliding window adds "and not the
distant past": $j > i - w$. Together, query $i$ sees the keys

$$
\max(0,\; i - w + 1),\; \dots,\; i ,
$$

which is $w$ keys once $i \ge w - 1$ and $i + 1$ keys before that. As a matrix, the
mask is a band of width $w$ lying on and below the diagonal.

The cleanest way to build any such mask is to stop thinking about triangles and write
the rule as a comparison of two index grids. A column vector of query indices and a
row vector of key indices broadcast to every pair $(i, j)$, and the rule is typed in
exactly as it is stated. There is nothing to get wrong about offsets of `triu` and
`tril`, and the limit cases need no code: for $w \ge T$ the second condition is always
true and the mask is causal; for $w = 1$ only the diagonal remains.

Attention under the mask is unchanged from the causal case. The hidden scores are set
to $-\infty$ before the softmax, so they contribute $e^{-\infty} = 0$ to the numerator
and to the denominator: each row is a proper distribution over its visible keys, and
the output of query $i$ is a function of $q_i$ and of the keys and values inside its
window only. Every row keeps its diagonal, so no row is entirely $-\infty$.

## Reference solution (PyTorch)

```python
def window_mask(T, window):
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (j > i - window)


def sliding_window_attention(q, k, v, window):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    visible = window_mask(q.shape[-2], window)
    scores = scores.masked_fill(~visible, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

JAX and MLX use `where(visible, scores, -inf)`. The same band can be written as
`tril(ones) & triu(ones, diagonal=-(window - 1))`; note the `window - 1`.

## Common mistakes

- **Window off by one.** `j >= i - window`, or `triu(..., diagonal=-window)`, keeps
  `window + 1` keys. The mask looks right, and with a window in the thousands nobody
  counts. This is not hypothetical: descriptions and code bases differ on whether the
  window includes the current token. The Mistral 7B announcement describes token $i$
  attending to tokens $[i - \text{sliding\_window},\, i]$, an interval that read
  literally contains one more token than the rule used here and in Gemma. When you
  port a model, find out which one its weights were trained with.
- **Symmetric window.** `abs(i - j) < window` is the mask of a bidirectional local
  encoder. In a decoder it lets each token see the next `window - 1` tokens, which do
  not exist yet at generation time. Earlier outputs then change when later tokens are
  appended, and a KV cache, which never recomputes them, disagrees with the full
  forward pass.
- **Window measured from the end of the sequence.** "The last `window` tokens" is the
  right set of keys for the last query, which is the only one a single decoding step
  computes. Used for all rows (by slicing `k[..., -window:, :]`, or with `j >= T -
  window`), it hides everything from the early queries, whose rows become all
  $-\infty$ and whose outputs are NaN, and shows the middle ones the wrong keys.
- **The opposite mask convention.** Half of the libraries use `True` for "attend" and
  the other half for "hide". A function that returns a mask must say which, and this
  one says `True` means visible.
- **Masking after the softmax.** Zeroing the weights outside the window leaves the
  visible weights divided by a sum over the whole causal row. They no longer sum to 1,
  and they still depend on the hidden keys.

## Complexity

As written, $O(B\,H\,T^2\,d)$ time and $O(B\,H\,T^2)$ memory, the same as full
attention: the masked scores are computed and thrown away. Only the band of
$O(T\,w)$ scores is needed, and a real implementation computes only those.

## In the wild

During generation nobody builds this mask. A decoding step has one query, the newest
token, and it must see the last `window` keys. If the cache of a windowed layer keeps
exactly the most recent `window` keys and values (a rolling buffer that overwrites the
oldest entry), the single row of scores needs no mask at all, and the layer's memory
is constant however long the conversation gets. The mask is needed for prefill, where
all prompt positions are processed in one pass, and it must agree exactly with what
the rolling cache will later hold. That is the reason the off-by-one matters.

A window does not limit how far information travels through the model. Each windowed
layer moves it at most $w - 1$ positions, so after $L$ such layers a token can be
influenced by tokens up to $L\,(w - 1)$ positions back.

Mistral 7B v0.1 used a window of 4,096 in every layer; the v0.2 release dropped
sliding-window attention in favor of a full 32k context. Gemma 3 mixes the two kinds
of layer, five local layers with a span of 1,024 tokens for each global layer, which
keeps most of the KV cache bounded while the global layers preserve long-range access.
In Google's PyTorch reference the local mask is the causal mask plus a large negative
number on `tril(..., diagonal=-sliding_window_size)`, that is on $j \le i - w$, and in
Hugging Face Transformers the overlay is `kv_idx > q_idx - sliding_window`. Both are
the rule of this problem.
