## The idea

The block is two lines of algebra, and you have written both parts before: the Gated
DeltaNet mixer in `gated-delta-layer` and SwiGLU in `swiglu-mlp`. What is new is what
the checkpoint does not say.

**Two conventions for a norm weight, in one block.** In the reference implementation
(`Qwen3_5RMSNorm` in Hugging Face Transformers 5.23) the block norm's parameter is
initialized to zeros and applied as `x_normed * (1.0 + weight)`. The norm inside the
delta layer (`Qwen3_5RMSNormGated`) initializes its parameter to ones and applies it as
`weight * x_normed`. Both are called `...norm.weight` or `...layernorm.weight` in the
file, and both have a shape that fits either reading. The values tell them apart: in
layer 1 of Qwen3.5 0.8B the two block norms have means 0.43 and 0.17, and the gated
norm 0.93. A gain belongs near one.

| tensor | stored | used as |
|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | gain − 1 | `1 + w` |
| `linear_attn.norm.weight` | gain | `w` |

The final norm of the model and the query and key norms of its attention layers are
offsets as well; they are the subject of `real-hybrid-logits`.

**An epsilon that only real activations test.** RMSNorm divides by
$\sqrt{\operatorname{mean}(u^2) + \epsilon}$. On a standard normal test vector the mean
square is 1 and any $\epsilon$ below $10^{-4}$ is float noise. The hidden states of this
model have a typical entry of 0.02 to 0.04, a mean square near $10^{-3}$, and there
$10^{-5}$ in place of $10^{-6}$ changes the norm by about half a percent. The config
says `rms_norm_eps: 1e-06`; the kit's `ops.rms_norm` defaults to `1e-5`, the value of
the Granite models.

**The state.** A linear-attention layer cannot rebuild its past from a cache of keys
and values. It carries the last $K - 1 = 3$ rows of the projected input (6144 numbers
each) for the convolution and the memory $S$ of the delta rule (16 heads of
$128 \times 128$): 280,576 numbers per layer, whatever the length of the sequence.

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

The JAX and MLX versions are the same lines. On the fixture's prompt this agrees with
`transformers` to within $3 \cdot 10^{-7}$ at the compared columns (entries of size
0.03), in all three frameworks and whether the 12 rows go in one call or two, although `transformers` computes a multi-token call with the chunked form
of the delta rule and this is the token-by-token form: the two are the same recurrence.

## Common mistakes

- **The stored block-norm weight used as a gain.** The stream is scaled by about 0.4
  going into the mixer and 0.2 going into the MLP. Nothing overflows and nothing looks
  broken; `mlx-lm` avoids the question by adding the one when it loads the file.
- **The one added to the gated norm too.** "This model's norms are offsets" is true of
  five kinds of norm weight (the two block norms, the final norm, `q_norm`, `k_norm`)
  and false of this one.
- **The default epsilon.** Right on every random test tensor, half a percent off on
  the real model.
- **`dt_bias` left out**, or `A_log`. The first row of a new sequence is unchanged, so
  a single-token smoke test passes.
- **A flipped kernel.** `conv1d.weight` is applied as stored, the last tap on the
  current row. A hand-written "convolution" that reverses it reads the past in the
  wrong order.
- **Gate before norm.** A Mamba-2 block normalizes `y * silu(z)`; this layer
  normalizes `y` and then multiplies.
- **The MLP reading the block's input** instead of the stream after the mixer.
- **Gate and up exchanged** in SwiGLU: the two matrices have the same shape.
- **Computing in bfloat16** because the file is bfloat16. The weights are decoded to
  float32 for a reason: eight bits of mantissa fail this tolerance on the small model
  already.
- **State dropped or half carried.** Ignoring `state` breaks every continued call;
  carrying `S` and restarting the convolution breaks the first three rows after each
  boundary, which is exactly the prefill-then-decode pattern of a server.
- **The state updated in place.** `memory *= g` overwrites the `S` the caller handed
  in. A server keeps a snapshot of the state to roll back a rejected speculative
  token, to branch a beam, or to reuse a cached prefix for two requests; after an
  in-place update the snapshot is the new state and the old one is gone. Every
  split-sequence test passes, because each call receives a fresh tensor.
- **Names matched without the trailing dot.** `"model.language_model.layers.1"` is a
  prefix of layers 10 to 19, and this file lists `layers.1.` before `layers.10.`, so a
  dict built from every matching name ends up with another layer's tensors. Correct for
  layer 13, and for every model of at most ten layers.
- **Calling `ops.gated_delta_layer`.** It is the answer to the previous problem.

## Complexity

Per call $O(T \cdot D \cdot (C + F))$ for the projections and the MLP and
$O(T \cdot H \cdot d_v \cdot d_k)$ for the recurrence: for the real sizes about 22
million multiply-adds per token in the block, under a twentieth of them in the delta
rule. The
state is $3C + H d_v d_k$ numbers per layer and sequence and does not grow.

## In the wild

`transformers` keeps the same state in another layout (four rows of convolution input
as `(1, 6144, 4)`, and the memory transposed, keys before values), and it switches
between a chunked rule for prompts and this recurrent rule for one cached token; its
two paths agree with each other to about $10^{-4}$ on this model's logits, measured for
this curriculum. `mlx-lm` 0.32.2 adds the `1.0` to the offset norms once, at load time,
and does it in bfloat16, which rounds every gain; the consequence for the logits is in
the editorial of `real-hybrid-logits`. The layer itself is from "Gated Delta Networks:
Improving Mamba2 with Delta Rule" (arXiv 2412.06464).
