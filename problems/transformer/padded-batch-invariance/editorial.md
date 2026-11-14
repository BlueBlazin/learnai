## The idea

Batching must be invisible: row $b$ of the batched result has to equal the model run
on sequence $b$ alone. Look at the last real token of a short row and ask what could
make the batched computation differ. There are exactly three channels.

1. **Attention.** The token's output is a weighted sum over the keys it may see. Alone,
   those are its own sequence's tokens up to itself. In the batch, the padding slots
   are also in the row, so they must be removed from every real query's view: the
   padding condition goes on the **key** axis. Combined with causality, query $i$ of
   row $b$ sees key $j$ when $j \le i$ and $j \ge P_b$.
2. **Position.** Alone, the first token is at position 0. Each row counts from its
   own first real token: $\text{position} = j - P_b$.
3. **Readout.** The answer is the logits at the last real token. Left padding puts it
   in column $T - 1$ for every row; right padding puts it in column $L_b - 1$, a
   different column per row.

Induction over the layers finishes the argument: if the real tokens of a row hold the
same vectors as in the solo run at the input of a layer, and attention only reads real
keys with the same relative positions, they hold the same vectors at its output.
What the padding slots hold never matters, provided it is a finite number.

That proviso is the one numerical trap. With left padding, a padding query at column
$i < P_b$ has no key it may attend to. A softmax over a row that is entirely $-\infty$
is $0/0$ = NaN, so the padding slot's vector becomes NaN. In the next layer the real
queries give that slot an attention weight of exactly 0, and $0 \cdot \text{NaN}$ is
NaN: the padding has leaked after all, through the values. The kit's `ops.attention`
returns zero for a query with nothing to see, which is why the provided block is safe.

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

Slicing the last column before the final norm and the head saves the projection to
the vocabulary for every other position; the norm and the head act on each position
independently, so the result is the same.

## Common mistakes

- **Attending to padding.** Using the causal mask alone. Every short row then computes
  the logits of a longer sequence that starts with pad tokens. The result looks
  reasonable and depends on the batch.
- **The padding mask on the wrong axis.** `real[:, None, :, None]` blocks padding
  *queries* (whose outputs nobody reads) and leaves padding *keys* visible to
  everyone. The shapes broadcast identically. The mask's last axis is the key.
- **Replacing the causal mask instead of combining.** A mask that only hides padding
  lets every token see the real tokens after it.
- **Finding padding by value.** `tokens != pad_id` also hides real tokens that happen
  to have the padding id. With the common setup `pad_token = eos_token`, that is every
  end-of-sequence token inside a multi-turn prompt. Use the lengths (in practice: the
  attention mask the tokenizer returns), never the values.
- **Right padding with a last-column readout.** With right padding and a causal mask
  the real tokens are actually computed correctly without any padding mask, because
  the padding is in their future. But column $-1$ then holds a padding slot for every
  short row. Either read column $L_b - 1$ per row, or pad on the left.
- **NaN from fully masked rows.** Described above. It passes in a one-layer model,
  where nothing reads the poisoned slots, and fails from two layers on. The usual
  fixes are a softmax that maps an empty row to zeros, a large finite negative number
  instead of $-\infty$, or letting padding queries attend to themselves (the rule the
  `attention-mask-builder` problem uses).

**About the positions.** Step 3 of the specification is the right habit, and it is
what you need for learned absolute position embeddings. For this model, though, it
cannot change the logits: RoPE scores depend only on the *difference* between the
query's and the key's positions, so shifting all positions of a row by the same $P_b$
(which is what counting from the padded start does) gives the same attention weights,
up to float32 rounding. The tests therefore cannot tell `column - pads` from `column`,
and neither can the model. The per-row positions start to matter the moment tokens
are appended with a KV cache: the cached keys were rotated with one set of positions,
and the next token must continue that same numbering.

## Complexity

One pass over a `(B, T)` tensor with $T = \max_b L_b$: $O(L\,B\,T\,D\,(D + M))$ for
projections and MLPs and $O(L\,B\,H\,T^2 d)$ for attention, plus $O(B\,D\,V)$ for the
head on the last column only. The padding is computed and thrown away: a batch of one
long and many short sequences wastes most of its arithmetic, which is why engines sort
or bucket requests by length before batching them.

## In the wild

Hugging Face `generate` expects `tokenizer.padding_side = "left"` for decoder-only
models and warns when it detects right padding, for exactly the readout reason above.
Its `attention_mask` of shape `(B, T)` is expanded to the `(B, 1, T, T)` additive mask
built here, and for the fully masked rows of left padding Transformers has a helper,
`AttentionMaskConverter._unmask_unattended`, that lets those rows attend to
everything rather than produce NaN in the fused attention kernel. For models with
learned absolute positions (GPT-2), `generate` derives `position_ids` from the mask
with a cumulative sum; for RoPE models such as Llama the positions are a plain
`arange`, which works because of the shift invariance described above. Engines built
on continuous batching (vLLM, llama.cpp's server) mostly avoid padding altogether in
the decode phase: every running sequence contributes exactly one new token per step,
and each attends to its own cache.
