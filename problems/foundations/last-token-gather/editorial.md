## The idea

This problem splits into two parts, and most bugs come from getting only one of them right.

**Where is the last real token?** In a batch of `B` sequences padded to length `T`, a row with `n` real tokens padded on the right places them at positions $0, \dots, n - 1$, so the last real token is at $n - 1$. When padded on the left, those tokens sit at $T - n, \dots, T - 1$, so the last real token is always at $T - 1$ regardless of `n`.

**How do you select a different position in every row?** A standard slice selects the same positions in every row, so it cannot handle right padding. Instead, use paired (advanced) indexing with two integer tensors of shape `(B,)`, one for the row indices and one for the target positions. Together, they select a list of `(row, position)` pairs.

## Reference solution (PyTorch)

```python
def last_token_logits(logits, lengths, padding_side):
    batch, seq, _ = logits.shape
    if padding_side == "right":
        last = lengths - 1
    elif padding_side == "left":
        last = torch.full_like(lengths, seq - 1)
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    rows = torch.arange(batch, device=logits.device)
    return logits[rows, last]
```

JAX and MLX support the same `logits[rows, last]` indexing. You can also express this with an explicit gather along axis 1:

```python
index = last[:, None, None].expand(batch, 1, logits.shape[-1])    # (B, 1, V)
torch.gather(logits, 1, index)[:, 0]                               # (B, V)
```

`torch.gather` along axis 1 computes `out[b, 0, v] = logits[b, index[b, 0, v], v]`, and PyTorch requires `index` to match the full `V` axis. By contrast, `torch.take_along_dim`, `jnp.take_along_axis`, and `mx.take_along_axis` broadcast a `(B, 1, 1)` index automatically.

## Common mistakes

- **Always using `logits[:, -1]`.** This works for a single prompt, equal-length batches, and left padding, but on a right-padded batch it reads a padding position for every short row.
- **Using `lengths` directly as the index.** A length is a count, so the last index is `lengths - 1`. Indexing with `logits[rows, lengths]` reads the first padding position and goes out of bounds on a full row. PyTorch raises an error, while JAX silently clamps the out-of-bounds index to `T - 1`, making full rows appear correct.
- **Ignoring `padding_side`.** Using `lengths - 1` on a left-padded batch points into the padding or into the middle of the prompt. Another common mistake is returning $T - n$, which selects the *first* real token instead of the last.
- **Using one position for the whole batch.** Indexing with `lengths.max() - 1` reads the last column of the longest row, so it only passes when all rows have the same length.
- **Indexing the wrong axis.** Writing `logits[last, rows]` pairs sequence positions with the batch axis and row numbers with the sequence axis. Whenever those integers happen to be in range, it runs without error and returns logits from other rows. Gathering along the vocabulary axis returns shape `(B, T)`, which only fails a shape check when $T \ne V$.
- **Multiplying by a one-hot mask and summing.** Computing `(logits * onehot[:, :, None]).sum(1)` does $O(T)$ times more work and fails whenever any unselected entry is `-inf`, because `0 * -inf` is `NaN`. Since banned tokens are represented as `-inf`, this happens in normal use. If you must use a mask, select with `where`.

## Complexity

$O(B\,V)$ time and memory, since one length-`V` vector is copied per row. The one-hot multiply-and-sum approach reads all $B\,T\,V$ entries instead.

## In the wild

Decoder-only generation code usually avoids this gather by padding on the left, as recommended by the Hugging Face generation guide. Because the model was never trained to continue from padding, left padding keeps `logits[:, -1]` at the last real token on every decoding step as new tokens are appended on the right. However, left padding shifts the work elsewhere. Position IDs must start at the first real token rather than column 0, and the attention mask must hide the padded prefix.

The gather is still required wherever right padding is standard, such as last-token pooling for sentence embeddings, scoring right-padded continuations, reading out a classifier token, and training-style evaluation pipelines. Engines with continuous batching avoid padding altogether by tracking each sequence's length separately and running only the newest token of each active sequence.
