## The idea

For every batch row `b`, query position `i`, and key position `j`, the mask decides whether query `i` may attend to key `j`. Two independent rules disallow attention:

- **Causal**: the key is in the future ($j > i$). This depends only on `(i, j)` and not on the batch row, giving a `(T, T)` lower-triangular pattern.
- **Padding**: the key is not a real token. This depends only on `(b, j)` and not on the query, giving a `(B, T)` pattern along the **key** axis.

Padding belongs on the key axis because attention reads from keys and values. Hiding a padding key removes its value vector from every weighted sum. Hiding a padding *query* only changes what a padding position computes for itself, which real tokens never read.

Combining the two rules gives `causal[i, j] AND real[b, j]`, which broadcasts `(1, T, T)` against `(B, 1, T)`.

**The empty row.** With left padding, a query `i` in the padded prefix only sees keys $0, \dots, i$, which are all padding. Without an exception, that entire row is $-\infty$, and its softmax is $0 / 0$ (`NaN`). That `NaN` lands in a padding position's output, which looks harmless at first. In the next layer, however, that position becomes a key again. Even though it is masked out for real queries, $\text{NaN} + (-\infty)$ is still `NaN`, so the `NaN` propagates into every real token of the row. Allowing the diagonal (`j == i`) avoids empty rows. A padding position attends to itself, produces finite garbage, and contributes exactly zero to real tokens once masked.

**Right padding hides less than it seems.** With right padding, all padding lies in the future of every real token, so the causal rule already keeps real queries away from padding. The padding rule only affects the rows of padding queries. That is why code that omits the padding mask can pass right-padded tests and fail on the first left-padded batch.

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

JAX and MLX construct the same boolean masks and finish with `where(allowed, 0.0, -inf)`. Every step is a broadcast comparison on `arange(T)`, avoiding explicit indexing and Python loops.

## Common mistakes

- **Padding on the query axis.** Writing `real[:, :, None]` instead of `real[:, None, :]` hides the rows of padding queries while leaving padding key columns open. With right padding, the real query rows are unaffected, so this bug only surfaces with left padding.
- **Left and right exchanged.** With left padding, the real tokens occupy the *last* `lengths[b]` positions ($j \ge T - \text{lengths}[b]$). A batch with no padding cannot distinguish the two sides.
- **Rows that are entirely hidden.** Omitting the diagonal rule leaves all-`-inf` rows for left-padded sequences, which produces `NaN` after softmax.
- **Over-correcting for padding queries.** Allowing a padding query to attend to itself *only* produces a different mask from the specification. Both choices are harmless for real tokens, but an implementation must match the specified contract.
- **Using a large finite number instead of `-inf`.** `-1e9` is not representable in float16. The common alternative, the most negative finite float, overflows as soon as two masks are added together, and a fully hidden row silently softmaxes to a uniform distribution instead of signaling an error.
- **Off by one.** Using `j <= i + 1` (or `triu` with the wrong diagonal) lets every token see the next token. The model then looks artificially accurate during evaluation and degrades during generation, when the next token does not exist. Similarly, `j <= lengths[b]` lets the first right-padding position through.
- **Missing the head axis.** Adding a `(B, T, T)` mask to `(B, H, T, T)` scores either fails to broadcast or, when `B == H`, silently pairs batch rows with attention heads.

## Complexity

$O(B\,T^2)$ time and memory. The `(T, T)` causal mask can be built once and cached, leaving only the `(B, T)` padding mask to depend on the batch.

## In the wild

Hugging Face Transformers builds a mask with the same `(batch, 1, query_length, key_length)` shape by combining a causal pattern with the tokenizer's `attention_mask`. Its additive form uses the most negative finite float (`torch.finfo(dtype).min`) rather than `-inf`, so a fully hidden row softmaxes to a uniform distribution and stays finite. For fused attention kernels, it has also used a helper that detects fully hidden rows (such as the leading rows under left padding) and unmasks them completely. Opening the diagonal is a third way to handle the same empty-row problem.

Inference engines that generate one token at a time with a KV cache rarely materialize the full `(T, T)` mask. Because the new token is the final query and every cached token is in its past, they only need the key-padding mask of shape `(B, 1, 1, T)`. Engines with continuous batching avoid padding altogether by tracking each sequence's slot in the cache separately.
