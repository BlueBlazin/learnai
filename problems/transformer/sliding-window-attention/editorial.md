## The idea

Let $T$ be the sequence length and $w$ be the window size. A standard causal mask allows only past and current keys ($j \le i$), and a sliding window also requires keys to fall within the window ($j > i - w$). Combining both inequalities means query $i$ attends to the keys

$$
\max(0,\; i - w + 1),\; \dots,\; i .
$$

That gives $w$ visible keys whenever $i \ge w - 1$, and $i + 1$ keys near the start of the sequence. As a `(T, T)` matrix, the mask is a diagonal band of width $w$ on and below the main diagonal.

The clearest way to construct this mask is to compare index grids directly instead of manipulating triangle offsets. Broadcasting a `(T, 1)` column of query indices against a `(1, T)` row of key indices produces every pair $(i, j)$, letting you write the two inequalities verbatim. Edge cases require no special handling. When $w \ge T$, the lower bound $j > i - w$ is always true and the mask becomes standard causal attention. When $w = 1$, only the diagonal $j = i$ remains.

Once the mask is built, attention works the same way as in the causal case. Set hidden scores to $-\infty$ before the softmax so they contribute $e^{-\infty} = 0$ to both the numerator and the denominator. Every row then forms a valid probability distribution over its visible keys, and query $i$'s output depends only on $q_i$ and the keys and values inside its window. Because the diagonal is always visible, no row is all $-\infty$.

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

JAX and MLX replace `masked_fill` with `where(visible, scores, -inf)`. You can also express the same band with triangle helpers as `tril(ones) & triu(ones, diagonal=-(window - 1))`, paying attention to the `window - 1` offset.

## Common mistakes

- **Window off by one.** Writing `j >= i - window` or `triu(..., diagonal=-window)` keeps `window + 1` keys per row. The mask still looks like a plausible diagonal band, and when `window` is in the thousands nobody notices by eye. Descriptions and codebases also differ on whether the window count includes the current token. For example, the Mistral 7B announcement wrote the interval $[i - \text{sliding\_window},\, i]$, which literally contains one more token than the convention used here and in Gemma. Always verify which convention a checkpoint was trained with.
- **Symmetric window.** Writing `abs(i - j) < window` creates a bidirectional local encoder mask. In a decoder, it lets each token see the next `window - 1` future tokens, which do not exist during generation. Earlier outputs then change when new tokens are appended, causing cached decoding to disagree with a full forward pass.
- **Measuring the window from the end of the sequence.** Attending to "the last `window` tokens" (by slicing `k[..., -window:, :]` or checking `j >= T - window`) only works for the final query during single-token decoding. Applied to all rows, it masks out every key for early queries (producing all $-\infty$ rows and NaNs) and shows middle queries the wrong keys.
- **Flipping the boolean convention.** Some libraries use `True` to mean "attend" and others use `True` to mean "mask out." Here, `True` means visible.
- **Masking after the softmax.** Zeroing weights outside the window after taking the softmax leaves the visible weights normalized by the full causal denominator. They no longer sum to 1, and they still depend on the hidden keys.

## Complexity

This dense implementation takes $O(B\,H\,T^2\,d)$ time and $O(B\,H\,T^2)$ memory, the same as full causal attention, because masked scores are still computed before being discarded. Only the $O(T\,w)$ entries inside the band are actually needed, and production kernels compute only those.

## In the wild

During autoregressive decoding, inference engines do not build a `(T, T)` mask at all. A decoding step processes a single new query that must attend to the most recent `window` keys. If the layer's KV cache is a rolling buffer of size `window` that overwrites the oldest entry, that single query attends to the entire buffer with no mask and uses constant memory regardless of sequence length. The 2D mask is used during prefill, when all prompt positions run in one pass, and it must match the rolling cache's exact window so prefill and decoding stay consistent.

A local window does not limit total receptive field across depth. Each windowed layer passes information forward by up to $w - 1$ positions, so after $L$ windowed layers a token can depend on inputs up to $L\,(w - 1)$ positions earlier.

Mistral 7B v0.1 used a 4,096-token sliding window in every layer, while v0.2 replaced it with full 32k causal attention. Gemma 3 combines both by interleaving five 1,024-token local layers for every global layer, keeping most of the KV cache bounded while retaining long-range access. Google's PyTorch reference builds the local mask by adding a large negative value on `tril(..., diagonal=-sliding_window_size)` (positions $j \le i - w$), and Hugging Face Transformers uses `kv_idx > q_idx - sliding_window`. Both implement the exact rule in this problem.
