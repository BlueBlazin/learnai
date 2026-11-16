## The idea

Four problems compose into one runner, and the composition has a single rule: after
stage 1 the only source of truth is the checkpoint. Everything downstream is written
against three tensors per matrix (bytes, scales, biases), so the design question is
where the boundary between "format" and "model" runs. The answer that keeps every
stage short is two small accessors, and a forward pass that never touches a key of
the checkpoint except through them.

![The runner. Stages 1 and 4 turn float parameters into a checkpoint of packed codes, per-group scales and biases, and float gains. A step (stages 2 and 3) reads only the checkpoint: it dequantizes the embedding rows it looks up, multiplies by codes in every linear layer, and keeps rotated keys and values in the KV cache.](assets/runner.svg)

**Stage 1** is `groupwise-4bit` per matrix plus `bit-packing` in bytes: reshape a row
of codes to pairs and compute `low + 16 * high`. The result replaces the float
matrix under its own name, which is how MLX checkpoints are laid out, and it is what
makes "is the float matrix still there?" a question about bytes. Two sizes are worth
checking by hand before moving on: a 4-bit row has `in / 2` bytes, and the whole
checkpoint has `n * bits / 8 + 8 * n / group_size` bytes per matrix of `n` weights.

**Stage 2** is `decoder-forward` with every `x @ W.T` replaced by one function,
`qlinear(ckpt, name, x)`, which is `quantized-matmul` behind an unpack. The number of
bits is read from the shapes at the point of use, per matrix; a runner that reads it
once runs every uniform checkpoint and fails on the first mixed one, and mixed is
what stage 4 produces. The embedding is the one place where a float row is needed.
Index the bytes, the scales, and the biases with the token ids first, then
dequantize: `B * T` rows instead of `vocab_size`. With tied embeddings the same
codes are used twice, as a lookup at the bottom and as a `qlinear` at the top.
`ops.swiglu` takes float matrices, so the MLP is written out: two `qlinear`s, a
`silu`, a third `qlinear`.

**Stage 3** is `kv-cache-append` inside each layer, with the two things that problem
left out because it had no model around it. Positions: a new token's RoPE angle
comes from its absolute position `S + i`, so positions are `S + arange(T)`, never
`arange(T)`. Convention: the cache holds keys *after* rotation, so a cached key is
never rotated again and a cache built by anyone who follows the convention can be
continued by anyone else. That is what lets `generate_q` continue a prefix it has
never seen, and it is why the layout of the cache is part of the specification and
not an implementation detail. With `step_q` in place, `forward_q` is `step_q` with
no cache, and generation is one call for the prompt and one per token.

**Stage 4** generalizes the quantizer from a number of bits to a function from name
to bits, and adds the accounting. The recipe (8 bits for the embedding and the
head, 4 elsewhere) is the cheapest effective move in this model: the embedding is 5
percent of the weights and its error enters at every token and, tied, leaves through
every logit. The accounting must count what is stored: codes, two float32 numbers
per group, and the gains. Leave the metadata out and the recipe "fits" budgets it
exceeds by a fifth of a bit per weight.

## Reference solution (PyTorch)

```python
def _matrix(ckpt, name, group_size, rows=None):
    prefix = name[: -len("weight")]
    packed, scale, bias = ckpt[name], ckpt[prefix + "scales"], ckpt[prefix + "biases"]
    bits = 8 * packed.shape[-1] // (scale.shape[-1] * group_size)  # from the shapes
    if rows is not None:
        packed, scale, bias = packed[rows], scale[rows], bias[rows]
    return _unpack(packed, bits).to(torch.float32), scale, bias


def _qlinear(ckpt, name, x, group_size):
    codes, scale, bias = _matrix(ckpt, name, group_size)
    out_dim, n_groups = scale.shape
    xg = x.reshape(*x.shape[:-1], n_groups, group_size)
    cg = codes.reshape(out_dim, n_groups, group_size)
    dots = torch.einsum("...gk,ogk->...og", xg, cg)
    return (dots * scale).sum(dim=-1) + xg.sum(dim=-1) @ bias.T


def _embed(ckpt, tokens, group_size):
    codes, scale, bias = _matrix(ckpt, "model.embed_tokens.weight", group_size, rows=tokens)
    groups = codes.reshape(*tokens.shape, -1, group_size)
    return (groups * scale[..., None] + bias[..., None]).reshape(*tokens.shape, -1)


def step_q(ckpt, cfg, tokens, cache, group_size):
    batch, new = tokens.shape
    past = 0 if cache is None else cache[0][0].shape[2]
    positions = past + torch.arange(new)
    visible = torch.arange(past + new)[None, :] <= positions[:, None]
    ...
    for i in range(cfg.n_layers):
        ...
        q = ops.rope(q, positions, cfg.rope_base)
        k = ops.rope(k, positions, cfg.rope_base)  # keys are cached rotated
        if cache is not None:
            k = torch.cat((cache[i][0], k), dim=2)
            v = torch.cat((cache[i][1], v), dim=2)
        new_cache.append((k, v))
        attended = ops.attention(q, k, v, visible)
        ...
```

The full file adds the quantizer and packer of stage 1, the greedy loop, and the two
functions of stage 4. `torch.cat` builds new tensors, so the cache that was passed
in is untouched and can be continued twice.

Why the pieces agree: `_qlinear` computes exactly $x \hat W^\top$ (the identity of
`quantized-matmul` is an equality, not an approximation), and `_embed` computes
exactly rows of $\hat E$. So the runner *is* the float model with weights $\hat W$,
which is what the tests compare it with, and the only approximation in the whole
system is the rounding in stage 1.

## Common mistakes

- **A float matrix kept alongside the codes**, typically the embedding, "for the
  lookup". The checkpoint is then larger than the format, and the forward pass
  fails on any checkpoint somebody else wrote.
- **Nibble order.** Packing the first code into the high half of the byte
  round-trips with its own unpacker and is wrong for every other reader.
- **Codes not packed**: 4-bit codes stored one per byte make a "4-bit" checkpoint
  the size of an 8-bit one.
- **A constant group** divided by its range of zero: NaN codes. Give it scale 1.
- **One scale per row in the embedding lookup.** With a group size equal to
  `d_model` there is one group per row and the bug is invisible; with two groups per
  row the second half of every embedding is off.
- **Each bias times the sum of the whole input row.** The bias of a group
  multiplies the sum of *its* inputs.
- **Bits assumed, or read once for the whole checkpoint.** They are a property of
  each matrix.
- **Positions restart at 0 in every step.** Decoding one token at a time then
  gives every new token position 0.
- **No causal mask inside a chunk** when a cache is present. Invisible with one
  token per step; wrong when a prompt continues a cached prefix.
- **Keys cached before rotation** and rotated again each step. Self-consistent, but
  not the specified cache: it cannot continue a cache from another source, and it
  rotates `S + T` keys per step instead of `T`.
- **The cache bypassed**: recomputing the whole sequence for each token gives the
  right tokens at quadratic cost, and cannot continue a prefix it has no tokens for.
- **Metadata left out of the budget**, or **no budget check at all**: the recipe is
  returned although it does not fit.
- **The recipe ignored** (everything at 4 bits: under the budget, over the KL
  limit), or **an untied head left at 4 bits**.

## Complexity

Quantizing is $O(n)$ per matrix. A step over $T$ new tokens with $S$ cached costs
$O(T \cdot n)$ in the linear layers (the codes are read once per token) and
$O(T (S + T) d)$ per head in attention; generating $N$ tokens after a prompt of $P$
processes $P + N - 1$ tokens in total, against $O(N (P + N))$ without the cache.
Storage is `bits + 64 / group_size` bits per matrix weight with float32 metadata.
This reference unpacks a matrix each time it is used, which keeps the code short; a
kernel reads the packed bytes directly (the GPU kernel problems do that).

## In the wild

- MLX stores a quantized `Linear` or `Embedding` as packed codes under `weight`
  with `scales` and `biases` per group of 64 inputs, and multiplies with
  `mx.quantized_matmul`; its metadata is float16 in a float16 model, which is why
  "4-bit" is 4.5 bits per weight there and 5 here, where it is float32.
  `problems/quantization/quantize-real-weights` measures that on a real model.
- `mlx_lm.convert --quant-predicate` takes mixed recipes (`mixed_2_6`, `mixed_3_4`,
  `mixed_3_6`, `mixed_4_6`): `lm_head` always gets the higher width, and so do
  `v_proj` and `down_proj` in the first and last eighth of the layers and every
  third layer between. llama.cpp's GGUF presets likewise mix quantization types
  across tensors, and `mlx_lm`'s dynamic quantization chooses per-layer bits from
  measured sensitivities under a bits-per-weight target; `bit-allocation` solves
  that choice exactly.
- Keeping rotated keys in the cache is what RoPE implementations do, because the
  rotation of a key depends only on its own position.
