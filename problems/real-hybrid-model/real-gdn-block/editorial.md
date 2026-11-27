## The idea

The block combines two components from earlier problems: the Gated DeltaNet mixer (`gated-delta-layer`) and a SwiGLU MLP (`swiglu-mlp`). What makes porting a real checkpoint tricky is the conventions that live in code rather than in the safetensors metadata.

**Two norm conventions in the same block.** In Hugging Face Transformers 5.23, `Qwen3_5RMSNorm` initializes its block-norm weight to zeros and scales by `(1.0 + weight)`. Inside the delta mixer, `Qwen3_5RMSNormGated` initializes its weight to ones and scales by `weight` directly. Both tensors end in `norm.weight` and have shapes that work under either convention, so nothing fails at load time. Inspecting the values reveals the difference. In layer 1 of Qwen3.5 0.8B, `input_layernorm.weight` and `post_attention_layernorm.weight` have means `0.43` and `0.17`, while `linear_attn.norm.weight` has mean `0.93`. A plain gain sits near `1.0`.

| tensor | stored | used as |
|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | gain − 1 | `1 + w` |
| `linear_attn.norm.weight` | gain | `w` |

The model's final norm and the query/key norms in its full-attention layers also use the offset convention. Those appear in `real-hybrid-logits`.

**An epsilon that only real activations expose.** RMSNorm divides by $\sqrt{\operatorname{mean}(u^2) + \epsilon}$. On standard normal test inputs, $\operatorname{mean}(u^2) \approx 1$, so changing $\epsilon$ between `1e-5` and `1e-6` is hidden in float32 noise. In Qwen3.5 0.8B, hidden-state entries are around `0.02` to `0.04`, giving a mean square near $10^{-3}$. At that scale, using `1e-5` instead of `1e-6` shifts the normalized output by about 0.5%. The config specifies `rms_norm_eps = 1e-6`, whereas `ops.rms_norm` defaults to `1e-5` (matching Granite models).

**Fixed-size recurrent state.** Unlike full attention, a linear-attention layer does not cache past keys and values. It stores only the last $K - 1 = 3$ rows of the projected QKV input (`3 * 6144` floats) for the causal convolution and the delta-rule memory `S` (`16 * 128 * 128` floats). That totals 280,576 numbers per layer regardless of sequence length.

## Reference solution (PyTorch)

```python
NORM_EPS = 1e-6


def delta_mixer(h, weights, prefix, state):
    steps = h.shape[0]
    a_log, gain = weights[prefix + "A_log"], weights[prefix + "norm.weight"]
    heads, d_v = a_log.shape[0], gain.shape[0]
    value_dim = heads * d_v
    key_dim = (weights[prefix + "in_proj_qkv.weight"].shape[0] - value_dim) // 2
    d_k = key_dim // heads
    conv_state, memory = (None, None) if state is None else state
    if memory is None:
        memory = torch.zeros(heads, d_v, d_k)

    projected = h @ weights[prefix + "in_proj_qkv.weight"].T
    mixed, conv_state = ops.causal_conv1d(projected, weights[prefix + "conv1d.weight"], conv_state)
    mixed = F.silu(mixed)
    q = ops.l2norm(mixed[:, :key_dim].reshape(steps, heads, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, key_dim : 2 * key_dim].reshape(steps, heads, d_k))
    v = mixed[:, 2 * key_dim :].reshape(steps, heads, d_v)

    a = h @ weights[prefix + "in_proj_a.weight"].T + weights[prefix + "dt_bias"]
    g = torch.exp(-torch.exp(a_log) * ops.softplus(a))
    beta = torch.sigmoid(h @ weights[prefix + "in_proj_b.weight"].T)

    outputs = []
    for t in range(steps):
        memory = memory * g[t, :, None, None]
        error = v[t] - torch.einsum("hvk,hk->hv", memory, k[t])
        memory = memory + (beta[t, :, None] * error)[:, :, None] * k[t, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", memory, q[t]))
    y = torch.stack(outputs)

    z = (h @ weights[prefix + "in_proj_z.weight"].T).reshape(steps, heads, d_v)
    y = ops.rms_norm(y, gain, NORM_EPS) * F.silu(z)        # a plain gain; norm, then gate
    out = y.reshape(steps, value_dim) @ weights[prefix + "out_proj.weight"].T
    return out, (conv_state, memory)


def qwen_gdn_block(x, weights, layer, state):
    p = f"model.language_model.layers.{layer}."
    h = ops.rms_norm(x, 1.0 + weights[p + "input_layernorm.weight"], NORM_EPS)
    mixed, state = delta_mixer(h, weights, p + "linear_attn.", state)
    x = x + mixed
    h = ops.rms_norm(x, 1.0 + weights[p + "post_attention_layernorm.weight"], NORM_EPS)
    x = x + ops.swiglu(h, weights[p + "mlp.gate_proj.weight"],
                       weights[p + "mlp.up_proj.weight"], weights[p + "mlp.down_proj.weight"])
    return x, state
```

The JAX and MLX solutions follow the same steps. On the 12-token prompt, this matches `transformers` within $3 \times 10^{-7}$ across the compared columns (entries around `0.03`) in all three frameworks, whether run in one call or split across two. Even though `transformers` uses a chunked formulation for multi-token prompts and this code runs token by token, both compute the same recurrence.

## Common mistakes

- **Using stored block-norm weights as plain gains.** This scales the stream by roughly `0.4` into the mixer and `0.2` into the MLP without overflowing or raising errors. (`mlx-lm` avoids this at runtime by adding `1` when loading weights.)
- **Adding `1` to `linear_attn.norm.weight` as well.** Five norm types in Qwen3.5 use offsets (the two block norms, the final norm, `q_norm`, and `k_norm`), but the gated norm inside Gated DeltaNet does not.
- **Using the default RMSNorm epsilon (`1e-5`).** This passes on unit-variance random inputs and is off by about 0.5% on the real model.
- **Omitting `dt_bias` or `A_log`.** Row 0 of a fresh sequence has a zero memory state, so a single-token smoke test misses decay bugs.
- **Reversing the convolution kernel.** `conv1d.weight` is applied as stored with its last tap on the current token. Flipping it reads past tokens in reverse order.
- **Applying the gate before the norm.** Mamba-2 normalizes `y * silu(z)`, whereas this layer normalizes `y` first and multiplies by `silu(z)` afterward.
- **Feeding the original block input to the MLP.** The MLP must read the residual stream after adding the mixer output.
- **Swapping `gate_proj` and `up_proj` in SwiGLU.** Both matrices have shape `(F, D)`, so swapping them produces no shape error.
- **Computing in bfloat16.** Checkpoint weights are stored in bfloat16 and decoded to float32 by `weights`. Running the block in bfloat16 fails the tolerance even on the small model.
- **Dropping or partially carrying state.** Ignoring `state` breaks every continued call. Carrying `S` without `conv_state` breaks the first `K - 1 = 3` rows after any chunk boundary, which is the exact prefill-to-decode transition in serving.
- **Updating `S` in place (`memory *= g`).** Mutating the caller's state corrupts snapshots kept for speculative-decoding rollback, beam branching, or shared prefix caching. Simple split-sequence tests do not catch this if each call passes a fresh tensor.
- **Matching layer prefixes without the trailing dot.** `"model.language_model.layers.1"` is a prefix of layers 10 through 19. Because safetensors lists `layers.1.` before `layers.10.`, collecting matching keys into a dict overwrites layer 1 with later layers while leaving layer 13 and small models unaffected.
- **Calling `ops.gated_delta_layer`.** That helper implements the previous problem and is disallowed here.

## Complexity

Each call performs $O(T \cdot D \cdot (C + F))$ work in the linear projections and MLP, plus $O(T \cdot H \cdot d_v \cdot d_k)$ work in the recurrence. At Qwen3.5 0.8B dimensions, that is about 22 million multiply-adds per token, with less than 5% in the delta-rule recurrence. The carried state takes $3C + H d_v d_k$ floats per layer and sequence and stays constant with sequence length.

## In the wild

`transformers` stores the same state in a different layout: four convolution rows shaped `(1, 6144, 4)` and a transposed memory matrix with keys before values. It uses a chunked algorithm for prompts and this token-by-token recurrence for cached decoding. Measured on this model's logits, the two paths agree to about $10^{-4}$.

`mlx-lm` 0.32.2 adds `1.0` to the offset norms once at load time in bfloat16, which rounds every gain. The editorial for `real-hybrid-logits` examines how that affects logits. The Gated DeltaNet layer itself comes from "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464).
