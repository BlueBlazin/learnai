Write down the left-padded row for the shortest sequence in a batch. For its final real token, list every way the batched pass could differ from running that sequence alone: which keys it can attend to, what RoPE position it receives, and which column its logits are read from. Each item corresponds to one piece you need to construct.

---

Everything can be built from two small index tensors: `pads` (the number of leading pad slots per row, shape `(B, 1)`) and `column` (`arange(T)`, shape `(1, T)`).

- Compute `real = column >= pads` with shape `(B, T)`. Do not use `tokens != pad_id`, because a real token can equal `pad_id`.
- Build a boolean mask broadcastable to `(B, n_heads, T, T)`, where axis 2 is the query and axis 3 is the key. The causal condition depends on `(query, key)` and is shared across rows, while the padding condition depends on `(row, key)`. Reshape each with size-1 singleton axes where needed and combine them with logical AND.
- Compute `positions = column - pads` with shape `(B, T)`, which `decoder_block` accepts directly.
- Because the batch is left-padded, read every row's final hidden state from column `-1`.

A left-padding query has no valid keys to attend to. The provided `decoder_block` safely returns zero for such queries. If you write your own attention instead, remember that softmax over an all-`-inf` row produces NaN, which then leaks into real tokens in the next layer via `0 * NaN`.

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

Next, embed `tokens`, pass `x` through `decoder_block(cfg, layer_params(params, i), x, mask, positions)` for each layer `i`, slice `x[:, -1]`, apply the final RMSNorm with `cfg.norm_eps`, and multiply by the transposed output head (`model.embed_tokens.weight` if `cfg.tie_embeddings` is `True`, else `lm_head.weight`).
