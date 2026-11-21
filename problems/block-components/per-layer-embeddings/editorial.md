## The idea

A decoder normally sees a token's identity once, at the bottom: the embedding is the
input of layer 0, and every later layer has to carry what it needs of it in the
residual stream. Per-layer embeddings give each layer a fresh, layer-specific view of
the token. There are three pieces.

1. **A packed table.** Row `tokens[t]` of `table` is $L$ vectors of width
   $d_\text{ple}$ laid end to end. One gather returns all of them, and a
   `reshape(T, L, d_ple)` separates them, because consecutive slices of a row are
   exactly what a row-major reshape produces.
2. **A projected part.** The same shape is produced a second way, from the layer-0
   input: one linear map to $L\,d_\text{ple}$ channels, cut the same way, each slice
   RMS-normalized. The two parts are added and multiplied by $2^{-1/2}$, the factor
   that keeps the variance of a sum of two unit-variance terms at 1.
3. **A gated injection** at the end of every layer: project the stream down to
   $d_\text{ple}$, apply GELU, multiply by the layer's slice, project back up,
   normalize, add. The stream decides, channel by channel, how much of the slice to
   let in; the slice never enters unmodulated.

The three scales each do a separate job. $\sqrt{d_\text{ple}}$ is the usual embedding
scale for a table of that width. $2^{-1/2}$ rescales the sum. $D^{-1/2}$ undoes the
$\sqrt D$ that the main embedding was multiplied by, and it sits in front of a norm,
which divides any constant factor out again: it changes the result only through the
epsilon, by a relative $\epsilon\,(D - 1) / (2 \cdot \text{mean square of } h_0 W^\top)$.
For inputs of order 1 and $D = 16$ that is below $10^{-5}$ unless a slice happens to be
small. The case written for it uses inputs of order $10^{-3}$, where the mean square is
comparable to $10^{-6}$; the same case is the one that tells $\epsilon = 10^{-6}$ from
the kit's default $10^{-5}$ on every norm.

## Reference solution (PyTorch)

```python
EPS = 1e-6


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def ple_inputs(tokens, h0, table, w_model, norm_gain, n_layers):
    steps, hidden = h0.shape
    d_ple = table.shape[-1] // n_layers
    token_part = (table[tokens] * math.sqrt(d_ple)).reshape(steps, n_layers, d_ple)
    context = (h0 @ w_model.T) * hidden**-0.5
    context = ops.rms_norm(context.reshape(steps, n_layers, d_ple), norm_gain, eps=EPS)
    return (context + token_part) * 2.0**-0.5


def ple_inject(h, ple_i, w_gate, w_proj, post_gain):
    gated = gelu_tanh(h @ w_gate.T) * ple_i
    return h + ops.rms_norm(gated @ w_proj.T, post_gain, eps=EPS)


def ple_layer(params, x, ple, i, mask):
    u = ops.rms_norm(x, params["input_layernorm.weight"], eps=EPS)
    q = u @ params["self_attn.q_proj.weight"].T
    k = u @ params["self_attn.k_proj.weight"].T
    v = u @ params["self_attn.v_proj.weight"].T
    attn = ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T
    a = x + ops.rms_norm(attn, params["post_attention_layernorm.weight"], eps=EPS)

    u = ops.rms_norm(a, params["pre_feedforward_layernorm.weight"], eps=EPS)
    gate = gelu_tanh(u @ params["mlp.gate_proj.weight"].T)
    mlp = (gate * (u @ params["mlp.up_proj.weight"].T)) @ params["mlp.down_proj.weight"].T
    m = a + ops.rms_norm(mlp, params["post_feedforward_layernorm.weight"], eps=EPS)

    y = ple_inject(
        m,
        ple[:, i],
        params["per_layer_input_gate.weight"],
        params["per_layer_projection.weight"],
        params["post_per_layer_input_norm.weight"],
    )
    return y * params["layer_scalar"]
```

## Common mistakes

- **Cutting the row the other way.** `reshape(T, d_ple, L)` followed by a transpose has
  the right shape and hands layer $i$ every $L$-th column. With $L = d_\text{ple}$ even
  the transpose is not needed to get a legal shape.
- **The wrong scale on the table.** $\sqrt D$ is the scale of the main embedding; this
  table is $d_\text{ple}$ wide per layer and is scaled by $\sqrt{d_\text{ple}}$.
- **Leaving out the projected part,** or the $2^{-1/2}$ on the sum.
- **Dropping $D^{-1/2}$,** or using $\epsilon = 10^{-5}$. Both are invisible until the
  numbers are small (see above).
- **Layer 0's slice in every layer.** `ple[:, 0]` instead of `ple[:, i]`: correct for
  the first layer, and the reason a one-layer test proves nothing.
- **No activation on the gate,** or the exact (erf) GELU. The two GELUs differ by up
  to $5 \cdot 10^{-4}$, which a comparison at $10^{-4}$ sees.
- **Injecting in the wrong place:** between attention and the MLP, so the MLP reads a
  stream that already contains the slice; or normalizing the stream plus the branch
  instead of the branch.
- **`layer_scalar` before the injection,** which leaves the injected branch unscaled.
- **A plain pre-norm layer.** Gemma's layer normalizes each sublayer's output as well
  as its input, and its MLP gate is GELU, not SiLU.
- **`h += ...`** in the injection: it overwrites the caller's stream.

## Complexity

`ple_inputs` is one gather of $T$ rows of width $L\,d_\text{ple}$ and one matrix
product $T \times D \times L\,d_\text{ple}$, both once per token, not once per layer.
Each injection is two small products, $O(T\,D\,d_\text{ple})$. The table's size is
$V\,L\,d_\text{ple}$ numbers, of which a forward pass touches $T$ rows.

## In the wild

Checked on 2026-11-21 against the code installed with `transformers` 5.23.0
(`models/gemma4/modeling_gemma4.py`) and the `config.json` files on the Hugging Face
Hub. The Gemma 4 report was not read.

- **Configurations.** `google/gemma-4-E2B-it`: `hidden_size` 1536, 35 layers,
  `hidden_size_per_layer_input` 256, `vocab_size_per_layer_input` 262144,
  `hidden_activation` `gelu_pytorch_tanh`, `rms_norm_eps` 1e-6. `google/gemma-4-E4B-it`:
  `hidden_size` 2560, 42 layers, the same per-layer width and vocabulary. The tables
  are therefore 8960 and 10752 wide, 2.35 and 2.82 billion numbers.
- **`ple_inputs`** is `Gemma4TextModel.get_per_layer_inputs` (the table, an embedding
  with `embed_scale = hidden_size_per_layer_input ** 0.5`, reshaped to
  `(batch, seq, num_hidden_layers, hidden_size_per_layer_input)`) followed by
  `project_per_layer_inputs` (`per_layer_model_projection`, times
  `hidden_size ** -0.5`, the same reshape, `per_layer_projection_norm`, the sum times
  `2.0 ** -0.5`). The model's loop passes `per_layer_inputs[:, :, i, :]` to layer `i`.
- **`ple_inject` and `ple_layer`** are the end and the whole of
  `Gemma4TextDecoderLayer.forward`: `per_layer_input_gate`, the activation, the
  product with the slice, `per_layer_projection`, `post_per_layer_input_norm`, the
  add, and last `hidden_states *= self.layer_scalar`.

The public tests compare with those functions run on a tiny configuration
(`fixtures/make.py`). For `ple_inject` the layer's attention and MLP are replaced by
modules that return zeros; the script also asserts, with a constant in place of the
MLP, that the injection reads the stream after the MLP's add.

What this problem leaves out or simplifies:

- **Attention.** `Attn` here is single-head softmax attention from the kit. Gemma 4's
  attention has several heads with grouped keys and values, RMS norms on queries,
  keys, and values, rotary positions (see `partial-rope`), sliding windows on most
  layers, and layers that reuse another layer's keys and values. None of that touches
  the per-layer embedding. The fixture replaces `self_attn` by the same stand-in.
- **The expert branch.** With `enable_moe_block` the layer adds a routed expert branch
  to the MLP output before `post_feedforward_layernorm`. The E2B and E4B
  configurations both have `enable_moe_block: false`, so this problem's layer is
  theirs in that respect.
- **MLP width.** E2B has `use_double_wide_mlp: true`: the MLP of its KV-shared layers
  is twice as wide (E4B: `false`). Here the width is whatever `params` holds.
- **Batching.** The library's tensors have a leading batch axis.
- **`layer_scalar`** is a buffer that a new module fills with ones; the values in the
  shipped checkpoints were not read. The tests use values other than 1.
- **Inputs without token ids.** When the model is called with embeddings only, it
  recovers the ids by comparing against the embedding matrix; and
  `project_per_layer_inputs` called without the table part returns the normalized
  projection alone, without the $2^{-1/2}$. This problem always has ids.
- **`h0`** is passed in. In the model it is `embed_tokens(input_ids)`, already
  multiplied by $\sqrt D$.
- **KV-shared layers.** For the last `num_kv_shared_layers` layers the model may keep
  only the positions whose logits are needed, slicing the per-layer inputs with them.
