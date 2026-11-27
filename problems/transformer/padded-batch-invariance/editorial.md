## The idea

Batching must be invisible, so row $b$ of the batched output must match running the decoder on sequence $b$ alone. Let $L_b$ be the length of sequence $b$ in a batch of $B$ sequences, $T = \max_b L_b$ be the padded width, and $P_b = T - L_b$ be the number of leading pad slots in row $b$. Three channels could cause a padded row to diverge from a solo run:

1. **Attention.** A real token's attention output is a weighted sum over the keys it can see. Alone, it sees only earlier tokens in its own sequence. In a padded batch, padding slots sit in the same row, so they must be masked out on the **key** axis. Combining causality with the real-token condition allows query $i$ of row $b$ to attend to key $j$ if and only if $j \le i$ and $j \ge P_b$.
2. **Positions.** Alone, a sequence starts at position 0. Subtracting $P_b$ from each column index $j$ gives $\text{position} = j - P_b$, placing every row's first real token at position 0.
3. **Readout.** Next-token logits come from the last real token of each sequence. Left padding aligns that token at column $T - 1$ for every row, whereas right padding leaves it at a different column $L_b - 1$ per row.

By induction across layers, if every real token enters a layer with the same hidden vector as in a solo run, and attention reads only real keys with matching relative positions, every real token exits the layer with the same vector. The hidden states stored in padding slots do not matter as long as they stay finite.

Finiteness is the one numerical hazard. Under left padding, any padding query at column $i < P_b$ has no valid keys to attend to. Taking a softmax over an all-$-\infty$ row produces $0/0 =$ NaN, poisoning that padding slot's hidden state. In the next layer, real queries assign that slot an attention weight of 0, and $0 \cdot \text{NaN}$ still evaluates to NaN, leaking into the real tokens. The kit's `ops.attention` handles this by returning zero for any query with no unmasked keys, which keeps `decoder_block` safe.

## Reference solution (PyTorch)

```python
def batched_forward(params, cfg, seqs, pad_id):
    longest = max(len(seq) for seq in seqs)
    tokens = torch.tensor([[pad_id] * (longest - len(seq)) + list(seq) for seq in seqs])
    pads = torch.tensor([longest - len(seq) for seq in seqs])[:, None]  # (B, 1)
    column = torch.arange(longest)[None, :]  # (1, T)
    real = column >= pads  # (B, T)
    causal = column[0][None, :] <= column[0][:, None]  # (T, T)
    mask = causal[None, None, :, :] & real[:, None, None, :]  # (B, 1, T, T)
    positions = column - pads  # (B, T)

    x = params["model.embed_tokens.weight"][tokens]
    for i in range(cfg.n_layers):
        x = decoder_block(cfg, layer_params(params, i), x, mask, positions)
    x = ops.rms_norm(x[:, -1], params["model.norm.weight"], cfg.norm_eps)
    head = params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]
    return x @ head.T
```

Slicing `x[:, -1]` before the final RMSNorm and vocabulary projection avoids computing logits for the other $T - 1$ positions. Both operations act independently on each token position, so the result is unchanged.

## Common mistakes

- **Attending to padding.** Using only a causal mask lets every short row attend to its leading pad tokens, effectively running the model on a longer sequence prefixed with `pad_id`. The logits look plausible but depend on the rest of the batch.
- **Putting the padding mask on the wrong axis.** Using `real[:, None, :, None]` masks padding *queries* (whose outputs are discarded) while leaving padding *keys* visible to real queries. Both shapes broadcast cleanly to `(B, n_heads, T, T)`, so there is no shape error. Always place the padding mask on the last (key) axis.
- **Replacing the causal mask instead of combining.** Masking only padding lets tokens attend to future real tokens.
- **Detecting padding by token value.** Checking `tokens != pad_id` masks out legitimate tokens that happen to equal `pad_id`. When `pad_token = eos_token`, that hides every end-of-sequence token inside a multi-turn prompt. Always derive the mask from sequence lengths (or the tokenizer's attention mask).
- **Right padding with a last-column readout.** Under right padding with a causal mask, real tokens never see the trailing padding, so their hidden states are actually correct. However, column `-1` holds a pad token for every short row. You must either gather column $L_b - 1$ per row or pad on the left.
- **NaN from fully masked rows.** As noted above, an unprotected softmax on an all-$-\infty$ row produces NaN. A one-layer model hides this bug because nothing reads the poisoned padding slot afterwards, but two or more layers fail immediately. Standard fixes include zeroing out empty softmax rows, using a large finite negative mask value instead of $-\infty$, or letting padding queries attend to themselves (as in `attention-mask-builder`).

**A note on positions.** Computing `column - pads` is the right general practice and is required for models with learned absolute position embeddings. For a pure RoPE model, however, shifting every position in a row by the same offset $P_b$ leaves all relative differences $(i - j)$ unchanged, so plain `arange(T)` produces the same attention weights up to float32 rounding. Per-row positions still become essential as soon as you append tokens with a KV cache, because newly generated tokens must continue the exact position numbering used when caching the prompt keys.

## Complexity

Let $L$ be the number of layers, $B$ the batch size, $T = \max_b L_b$ the padded length, $D$ the model width, $M$ the MLP hidden width, $H$ the number of heads, $d$ the head dimension, and $V$ the vocabulary size. One batched pass takes $O(L\,B\,T\,D\,(D + M))$ time for linear projections and MLPs, $O(L\,B\,H\,T^2\,d)$ time for attention, and $O(B\,D\,V)$ time for the output head on the last column. Because padding slots are still computed and then discarded, batching one very long sequence with many short ones wastes most of its FLOPs. Serving engines mitigate this by sorting or bucketing requests by length.

## In the wild

Hugging Face's `generate` expects `tokenizer.padding_side = "left"` for decoder-only models and logs a warning if right padding is used, specifically so the final token aligns at column `-1`. Internally, it expands the `(B, T)` `attention_mask` into a `(B, 1, T, T)` additive mask, and uses `AttentionMaskConverter._unmask_unattended` to let fully masked left-padding rows attend to all positions rather than trigger NaNs in fused attention kernels.

For models with learned absolute positions like GPT-2, `generate` computes `position_ids` from the attention mask via a cumulative sum. For RoPE models like Llama, it can use a plain `arange` during prefill because of RoPE's shift invariance. Modern serving engines with continuous batching (such as vLLM and llama.cpp's server) largely eliminate padding during decode by stepping one new token per active sequence against each sequence's own KV cache.
