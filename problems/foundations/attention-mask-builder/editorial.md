## The idea

The mask answers one question for every (row, query, key) triple: may this query read
this key? Two independent rules say no.

- **Causal**: the key is in the future, $j > i$. This depends on `(i, j)` and not on
  the row: a `(T, T)` lower-triangular pattern.
- **Padding**: the key is not a real token. This depends on `(b, j)` and not on the
  query: a `(B, T)` pattern that enters along the **key** axis.

Padding is a property of the key because attention reads from keys and values. Hiding
a padding key removes its value vector from every weighted sum. Hiding a padding
*query* would only decide what a padding position computes for itself, which nobody
reads.

The combined boolean is `causal[i, j] AND real[b, j]`, built by broadcasting
`(1, T, T)` against `(B, 1, T)`.

**The empty row.** With left padding, query `i` in the padded prefix sees keys
$0..i$, all padding: the row is entirely hidden. The softmax of a row of `-inf` is
$0 / 0$. That NaN lands in the output of a padding position, which looks harmless, but
in the next layer that position is a key again. It is masked for real queries, and
$\text{NaN} + (-\infty)$ is still NaN, so the NaN reaches every real token of the row.
The specification avoids the empty row by always allowing the diagonal. A padding
position then attends to itself; its output is garbage but finite, and finite garbage
behind a mask contributes exactly zero.

**Right padding hides less than it seems.** With right padding the causal rule already
keeps real queries away from the padding, which lies entirely in their future. The
padding rule only changes the rows of padding queries. That is why code that forgets
the padding mask can pass every right-padded test and fail on the first left-padded
batch.

## Reference solution (PyTorch)

```python
def build_mask(lengths, T, padding_side):
    pos = torch.arange(T, device=lengths.device)
    if padding_side == "right":
        real = pos[None, :] < lengths[:, None]
    elif padding_side == "left":
        real = pos[None, :] >= T - lengths[:, None]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    causal = pos[None, :] <= pos[:, None]
    allowed = (causal[None, :, :] & real[:, None, :]) | (pos[None, :] == pos[:, None])
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=lengths.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[:, None, :, :]
```

JAX and MLX build the same booleans and finish with `where(allowed, 0.0, -inf)`.
Everything is a comparison of an `arange` with something, so there is no indexing to
get wrong and no loop.

## Common mistakes

- **Padding on the query axis.** `real[:, :, None]` instead of `real[:, None, :]`
  hides the rows of padding queries and leaves the padding columns open. For right
  padding the real rows are unchanged, so the bug surfaces only with left padding.
- **Left and right exchanged.** With left padding the real tokens are the *last*
  `lengths[b]` positions, $j \ge T - \text{lengths}[b]$. A batch without padding cannot
  tell the two apart.
- **Rows that are entirely hidden.** Leaving out the diagonal rule produces rows of
  `-inf` for every left-padded row, and NaN after the softmax.
- **Over-correcting for padding queries.** Letting a padding query attend to itself
  *only* is a different mask from the one specified here. Either choice is harmless
  for the real tokens; a specification has to pick one, and an engine has to match
  the reference it is compared against.
- **A large finite number instead of `-inf`.** `-1e9` works in float32 until it does
  not: it is not representable in float16 at all, the usual alternative (the most
  negative finite float) overflows as soon as two masks are added, and a fully hidden
  row gives a uniform distribution over hidden keys instead of an error anyone would
  notice.
- **Off by one.** `j <= i + 1` (or a `triu` with the wrong diagonal) shows every token
  the next token: the model appears to predict with uncanny accuracy in evaluation and
  degrades in generation, where the next token does not exist yet. `j <= lengths[b]`
  lets the first padding position through.
- **Missing the head axis.** A `(B, T, T)` mask added to `(B, H, T, T)` scores either
  fails to broadcast or, when `B == H`, silently pairs rows with heads.

## Complexity

$O(B\,T^2)$ time and memory. The `(T, T)` causal part can be built once and cached;
only the `(B, T)` padding part depends on the batch.

## In the wild

Hugging Face Transformers builds the same thing: a mask of shape
`(batch, 1, query_length, key_length)` that combines a causal pattern with the
tokenizer's `attention_mask`. Its additive form uses the most negative finite float
(`torch.finfo(dtype).min`) and not `-inf`, so a fully hidden row softmaxes to a uniform
distribution and stays finite. For the fused attention kernel it has also carried a
helper that finds fully hidden rows, "for example the relevant first rows when using
left padding", and opens them completely. This problem's rule (open the diagonal) is
a third answer to the same question; all three exist because the empty row has to go
somewhere.

Engines that generate one token at a time with a KV cache rarely build the `(T, T)`
mask at all: the new token is the last query, everything in the cache is in its past,
and only the key-padding part, one row of shape `(B, 1, 1, T)`, is needed. Engines
with continuous batching go further and avoid padding entirely by giving each
sequence its own slot in the cache.
