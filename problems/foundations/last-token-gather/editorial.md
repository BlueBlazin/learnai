## The idea

There are two separate questions, and the bugs come from answering only one of them.

**Where is the last real token?** A row with `n` real tokens padded on the right keeps
them at positions $0, \dots, n - 1$, so the last one is at $n - 1$. Padded on the left,
the real tokens are pushed against the end, $T - n, \dots, T - 1$, and the last one is
at $T - 1$ whatever `n` is.

**How do you select a different position in every row?** Slicing cannot do it: a slice
selects the same positions in every row. What is needed is paired (advanced) indexing:
two integer tensors of shape `(B,)`, one for the row and one for the position, read
together as a list of `(row, position)` pairs.

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

JAX and MLX accept the same `logits[rows, last]`. The gather form says the same thing
with an explicit axis:

```python
index = last[:, None, None].expand(batch, 1, logits.shape[-1])    # (B, 1, V)
torch.gather(logits, 1, index)[:, 0]                               # (B, V)
```

`gather` along axis 1 computes `out[b, 0, v] = logits[b, index[b, 0, v], v]`. The index
must have a full `V` axis in PyTorch; `take_along_dim`, `jnp.take_along_axis`, and
`mx.take_along_axis` broadcast a `(B, 1, 1)` index for you.

## Common mistakes

- **Always `logits[:, -1]`.** Correct for one prompt, for equal lengths, and for left
  padding. For a right-padded batch it returns the logits at a padding position for
  every short row.
- **`lengths` as the index.** A length is a count; the last index is one less.
  `logits[rows, lengths]` reads the first padding position, and for a full row it is
  out of range: an error in PyTorch, and silently clamped to the last position in JAX,
  where a full row then looks right.
- **Ignoring the padding side.** `lengths - 1` on a left-padded batch points into the
  padding, or at a token in the middle of the prompt. A variant returns $T - n$, the
  *first* real token.
- **One position for the whole batch.** `lengths.max() - 1` is the last column of the
  longest row. Tests with equal lengths pass.
- **Indexing the wrong axis.** `logits[last, rows]` pairs positions with the batch axis
  and row numbers with the sequence axis; it runs without error whenever the numbers
  happen to be in range, and returns other rows' logits. Gathering along the
  vocabulary axis returns shape `(B, T)`, which is only caught by shape when
  $T \ne V$.
- **One-hot multiply and sum.** `(logits * onehot[:, :, None]).sum(1)` is a common way
  to avoid thinking about gather. It is $O(T)$ times more work, and it breaks on the
  first `-inf` logit: `0 * -inf` is NaN. Banned tokens are represented as `-inf`, so
  this is not a corner case. If a mask is unavoidable, select with `where`.

## Complexity

$O(B\,V)$ time and memory: one vector is copied per row. The one-hot variant reads all
$B\,T\,V$ entries.

## In the wild

Generation code sidesteps the gather by padding on the left, as the Hugging Face
generation guide instructs for decoder-only models: the model was never trained to
continue from padding, and with left padding `logits[:, -1]` is the last real token at
every step of the loop, including after new tokens are appended on the right. Left
padding moves the difficulty elsewhere: position ids must be counted from the first
real token and not from column 0, and the attention mask must hide the padded prefix.

The gather is needed wherever right padding is the convention: computing sentence
embeddings by last-token pooling, scoring a batch of right-padded continuations,
reading out a classifier token, and training-style evaluation pipelines. Engines with
continuous batching avoid the question, since each sequence in the batch tracks its own
length and only the newest token of each is ever run through the model.
