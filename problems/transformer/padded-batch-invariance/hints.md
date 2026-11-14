Take the shortest sequence in a batch and write down its padded row. For its last
token, list every way in which the batched computation could differ from running that
sequence alone: which other vectors can reach it through attention, what position it
is rotated with, and which column its logits are read from. Each item on that list is
one thing to build.

---

Everything follows from one small tensor: `pads`, the number of padding slots in front
of each row, with shape `(B, 1)`, compared against the column index `arange(T)` with
shape `(1, T)`.

- `real = column >= pads` has shape `(B, T)`. Do not compute it as
  `tokens != pad_id`: a real token may have the same id.
- The mask must broadcast to `(B, n_heads, T, T)`, where the third axis is the query
  and the last axis is the key. The causal part depends on (query, key) and is the
  same for all rows; the padding part depends on (row, key). Give each a shape with
  ones in the axes it does not depend on, and combine them with a logical and.
- `positions = column - pads` has shape `(B, T)`, which `decoder_block` accepts.
- With left padding the answer of every row is in column `-1`.

A padding query ends up with no key it may attend to. The provided block returns zero
for it; if you write your own attention, a softmax over a row of `-inf` is NaN, and
that NaN reaches the real tokens in the next layer through `0 * NaN`.

---

In PyTorch:

```python
longest = max(len(seq) for seq in seqs)
tokens = torch.tensor([[pad_id] * (longest - len(seq)) + list(seq) for seq in seqs])
pads = torch.tensor([longest - len(seq) for seq in seqs])[:, None]   # (B, 1)
column = torch.arange(longest)[None, :]                              # (1, T)
real = column >= pads                                                # (B, T)
causal = column[0][None, :] <= column[0][:, None]                    # (T, T): key <= query
mask = causal[None, None, :, :] & real[:, None, None, :]             # (B, 1, T, T)
positions = column - pads                                            # (B, T)
```

Then embed `tokens`, run `decoder_block(cfg, layer_params(params, i), x, mask,
positions)` for every layer, take `x[:, -1]`, apply the final norm with
`cfg.norm_eps`, and multiply by the transposed head (the embedding matrix if
`cfg.tie_embeddings`, else `lm_head.weight`).
