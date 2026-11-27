## The idea

After stage 1, the quantized checkpoint is the only source of truth. Every downstream function operates on three tensors per matrix (packed bytes, scales, and biases). The cleanest way to separate the storage format from the transformer architecture is to write two small accessors, `_qlinear` and `_embed`, so the decoder forward pass never inspects checkpoint keys directly.

![Left, the checkpoint that stages 1 and 4 write. Right, one step of stages 2 and 3, which reads only that checkpoint.](assets/runner.svg)

**Stage 1** combines `groupwise-4bit` with `bit-packing`. For 4-bit weights, reshape each row of codes into pairs and pack them as `low + 16 * high`. Storing the packed uint8 array under the original `*.weight` key matches MLX's checkpoint layout and ensures no float weights remain behind. A 4-bit row uses `in / 2` bytes, and a matrix of $n$ weights uses `n * bits / 8 + 8 * n / group_size` bytes in total.

**Stage 2** adapts `decoder-forward` by replacing every linear projection `x @ W.T` with `_qlinear(ckpt, name, x, group_size)`. Derive `bits` from the tensor shapes inside the accessor for every matrix so mixed-precision checkpoints work automatically. For the embedding lookup, index `weight`, `scales`, and `biases` with `tokens` first and dequantize only those $B \cdot T$ rows instead of the whole vocabulary. When `cfg.tie_embeddings` is `True`, `model.embed_tokens.weight` serves as both the input embedding (via `_embed`) and the output head (via `_qlinear`). Because `ops.swiglu` expects float weight matrices, write the SwiGLU MLP explicitly using three `_qlinear` calls and `silu`.

**Stage 3** embeds `kv-cache-append` inside each decoder layer and enforces two cache invariants. First, every new token uses its absolute position `S + i` for RoPE (`past + arange(T)`), never `arange(T)`. Second, the cache stores keys *after* RoPE rotation so cached keys are never rotated twice and any cache built under the spec can be continued by `step_q` or `generate_q`. Once `step_q` works, `forward_q` is just `step_q(..., cache=None)[0]`, and `generate_q` prefills the prompt in one call followed by single-token decode calls.

**Stage 4** generalizes the quantizer to accept a per-tensor bit width and implements exact byte accounting. Keeping `model.embed_tokens.weight` (and `lm_head.weight` when untied) at 8 bits while quantizing everything else to 4 bits is the highest-leverage move in this model. The embedding accounts for only 5% of parameters, yet its quantization noise affects every input token and (when tied) every output logit. `bits_per_weight` must count every stored bit (packed codes, float32 scales and biases, and unquantized gains) so the budget check reflects true memory usage.

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

The complete reference adds the stage 1 quantizer and packer, the greedy loop in `generate_q`, and the stage 4 accounting and mixed-precision functions. Because `torch.cat` allocates new tensors, the caller's input `cache` is never modified.

`_qlinear` computes $x\,\hat W^{\top}$ exactly (the grouped dot-product identity is algebraic, not approximate), and `_embed` reconstructs exact rows of $\hat E$. The quantized runner therefore matches the float decoder evaluated on $\hat W$, and the only approximation in the entire pipeline is the rounding in stage 1.

## Common mistakes

- **Keeping a float copy of the embedding matrix.** Storing float weights alongside packed codes breaks the checkpoint size check and fails when the forward pass runs on a judge-built checkpoint.
- **Wrong nibble order.** Packing the first code into the high nibble round-trips with your own unpacker but fails against any checkpoint produced by the spec.
- **Storing 4-bit codes unpacked.** Saving one 4-bit code per uint8 byte doubles checkpoint size to match 8-bit storage.
- **Dividing constant groups by zero.** A group where max equals min has zero range and produces NaN codes unless its scale is set to `1.0`.
- **Using one scale per row in `_embed`.** This bug hides when `group_size == d_model` (one group per row) and corrupts embeddings whenever a row has multiple groups.
- **Multiplying each group bias by the whole input row sum.** Each group's bias must multiply only the sum of the inputs belonging to that group.
- **Hardcoding or caching a single bit width.** Bit width is a per-matrix property; reading it once breaks on mixed-precision checkpoints.
- **Resetting RoPE positions to 0 on each step.** Decoding one token at a time with `positions = arange(T)` assigns position 0 to every generated token.
- **Missing the causal mask inside a cached chunk.** Omitting the causal mask works for single-token decode steps ($T = 1$) but fails when a multi-token prompt continues an existing cache.
- **Caching unrotated keys.** Rotating cached keys on every step breaks interoperability with external caches and wastes work rotating $S + T$ keys instead of $T$.
- **Bypassing the KV cache in `generate_q`.** Recomputing the whole sequence on each token has quadratic attention cost and cannot continue a prefix whose tokens are only in `cache`.
- **Omitting metadata from `bits_per_weight` or skipping the budget check.** Ignoring scales and biases understates size, and skipping the check returns checkpoints that exceed `max_bpw`.
- **Ignoring the mixed recipe or leaving an untied `lm_head` at 4 bits.** Uniform 4-bit quantization stays under the size budget but exceeds the 0.14 nat KL divergence limit.

## Complexity

Quantizing a matrix of $n$ weights takes $O(n)$ time. One step over $T$ new tokens with $S$ cached tokens takes $O(T \cdot n)$ time in linear layers and $O(T(S + T)d)$ per head in attention. Generating $N$ tokens after a $P$-token prompt processes $P + N - 1$ tokens in total, compared to $O(N(P + N))$ without a cache. Storage is `bits + 64 / group_size` bits per matrix weight with float32 metadata. This reference unpacks a matrix to float32 on each use for clarity, whereas production GPU kernels read packed bytes directly.

## In the wild

- MLX stores quantized `Linear` and `Embedding` layers as packed `weight` bytes with per-group `scales` and `biases` (default `group_size = 64`) and multiplies via `mx.quantized_matmul`. Because MLX uses float16 metadata in float16 models, 4-bit quantization costs 4.5 bits per weight there versus 5.0 (plus norms) here with float32 metadata (`problems/quantization/quantize-real-weights` measures this on real weights).
- `mlx_lm.convert --quant-predicate` supports mixed recipes (`mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`) that give `lm_head` the higher bit width along with `v_proj` and `down_proj` in sensitive layers. `llama.cpp` GGUF presets similarly mix quantization types across tensors, and `mlx_lm`'s dynamic quantization allocates per-layer bit widths from measured sensitivities under a target budget (solved exactly in `bit-allocation`).
- Real RoPE serving stacks always cache post-rotation keys because a key's rotary embedding depends only on its own position.
