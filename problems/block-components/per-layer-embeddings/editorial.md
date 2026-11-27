## The idea

A standard decoder reads a token's embedding once at layer 0, forcing every later layer to recover token identity from the residual stream. Per-layer embeddings instead give every layer its own fresh, layer-specific view of the token. The mechanism has three parts.

1. **A packed embedding table.** Row `tokens[t]` of `table` stores $L$ vectors of width $d_{\text{ple}}$ laid end to end. One gather fetches all layers' vectors at once, and `reshape(T, L, d_ple)` separates them because consecutive slices of a row match row-major order.
2. **A projected context term.** The same `(T, L, d_ple)` shape is also computed from the layer-0 input `h0` by one linear projection to $L\,d_{\text{ple}}$ channels, reshaped the same way and RMS-normalized per slice. Adding the two terms and multiplying by $2^{-1/2}$ keeps the variance of a sum of two unit-variance terms at 1.
3. **A gated injection at the end of each layer.** The layer projects the residual stream down to $d_{\text{ple}}$, applies tanh-GELU, multiplies elementwise by the layer's slice, projects back up to $D$, normalizes the branch, and adds it to the stream. This lets the stream decide channel by channel how much of the token's per-layer signal to admit.

Each of the three scale factors has a distinct role. $\sqrt{d_{\text{ple}}}$ is the standard embedding scale for a table of width $d_{\text{ple}}$, and $2^{-1/2}$ rescales the sum of the two terms. $D^{-1/2}$ undoes the $\sqrt{D}$ scaling already present in `h0`. Because it sits immediately before an RMS norm, which divides out any constant scale factor, $D^{-1/2}$ affects the output only through $\epsilon$, by a relative amount of roughly $\epsilon\,(D - 1) / (2 \cdot \text{mean square of } h_0 W_{\text{model}}^\top)$. For order-1 inputs and $D = 16$, that shift is smaller than $10^{-5}$ unless a slice is small. When inputs have magnitude around $10^{-3}$, the mean square is comparable to $10^{-6}$, which exposes both a missing $D^{-1/2}$ and using $\epsilon = 10^{-5}$ instead of $10^{-6}$.

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

- **Reshaping in the wrong axis order.** Calling `reshape(T, d_ple, L)` followed by a transpose produces the right output shape `(T, L, d_ple)` but hands layer $i$ every $L$-th column instead of a contiguous slice. When $L = d_{\text{ple}}$, even without a transpose the shape looks valid.
- **Scaling the table by $\sqrt{D}$ instead of $\sqrt{d_{\text{ple}}}$.** $\sqrt{D}$ belongs to the main embedding table. Each per-layer slice has width $d_{\text{ple}}$ and must be scaled by $\sqrt{d_{\text{ple}}}$.
- **Omitting the projected context term or the $2^{-1/2}$ factor.** Both are required when combining the two branches in `ple_inputs`.
- **Dropping $D^{-1/2}$ or using $\epsilon = 10^{-5}$.** Both mistakes are masked on order-1 inputs and only show up when activations are small (around $10^{-3}$).
- **Hard-coding `ple[:, 0]` instead of `ple[:, i]`.** This passes any single-layer test and silently feeds layer 0's embeddings to every deeper layer.
- **Missing gate activation or using exact (erf) GELU.** Omitting GELU breaks the gating entirely. Using exact `erf` GELU instead of the tanh approximation differs by up to $5 \cdot 10^{-4}$, exceeding the `1e-4` relative tolerance.
- **Injecting in the wrong place or normalizing the residual sum.** Placing `ple_inject` between attention and the MLP changes the MLP's input. Applying `post_gain` RMS norm to `h + branch` instead of `branch` alone corrupts the residual stream.
- **Applying `layer_scalar` before `ple_inject`.** Scaling `m` instead of `y` leaves the injected branch unscaled.
- **Using a standard pre-norm block.** Gemma 4 normalizes both the input and the output of each sublayer, and uses GELU (not SiLU) for the MLP gate.
- **Mutating `h` in place.** Writing `h += ...` in `ple_inject` overwrites the caller's tensor.

## Complexity

`ple_inputs` performs one gather of $T$ rows of width $L\,d_{\text{ple}}$ and one matrix multiply of cost $O(T\,D\,L\,d_{\text{ple}})$, both paid once per forward pass rather than once per layer. Each layer's `ple_inject` adds two small matrix multiplies totaling $O(T\,D\,d_{\text{ple}})$. The table stores $V\,L\,d_{\text{ple}}$ parameters, of which each forward pass reads only $T$ rows.

## In the wild

Checked on 2026-11-21 against `transformers` 5.23.0 (`models/gemma4/modeling_gemma4.py`) and the `config.json` files on the Hugging Face Hub. The Gemma 4 technical report was not read.

- **Configurations.** `google/gemma-4-E2B-it` uses `hidden_size` 1536, 35 layers, `hidden_size_per_layer_input` 256, `vocab_size_per_layer_input` 262144, `hidden_activation: "gelu_pytorch_tanh"`, and `rms_norm_eps: 1e-6`. `google/gemma-4-E4B-it` uses `hidden_size` 2560 and 42 layers with the same per-layer width and vocabulary. Their packed tables therefore have widths 8960 and 10752, holding 2.35 and 2.82 billion parameters.
- **Mapping to `transformers`.** `ple_inputs` corresponds to `Gemma4TextModel.get_per_layer_inputs` (table lookup scaled by `hidden_size_per_layer_input ** 0.5` and reshaped to `(batch, seq, num_hidden_layers, hidden_size_per_layer_input)`) followed by `project_per_layer_inputs` (`per_layer_model_projection`, scaled by `hidden_size ** -0.5`, reshaped, normalized with `per_layer_projection_norm`, and added to the table term with a `2.0 ** -0.5` factor). The decoder loop passes `per_layer_inputs[:, :, i, :]` to layer `i`. `ple_inject` and `ple_layer` correspond to the end and full body of `Gemma4TextDecoderLayer.forward`, ending with `hidden_states *= self.layer_scalar`.

The public tests compare against those exact Hugging Face modules on a tiny configuration (`fixtures/make.py`). For `ple_inject`, the fixture replaces attention and the MLP with zero-returning modules, and also verifies with a constant MLP output that the injection reads the stream after the MLP's residual add.

Deliberate simplifications relative to the full `transformers` model:

- **Attention.** `Attn` here uses single-head softmax attention from `learnai.kit`. Gemma 4 uses multi-head grouped-query attention, RMS norms on queries, keys, and values, rotary positions (see `partial-rope`), sliding windows on most layers, and KV-sharing across layers. None of those interact with the per-layer embedding, and the fixture swaps in the same stand-in attention.
- **MoE branch.** When `enable_moe_block` is true, the layer adds a routed expert branch to the MLP output before `post_feedforward_layernorm`. Both E2B and E4B set `enable_moe_block: false`, so this problem matches their layer structure.
- **MLP width.** E2B sets `use_double_wide_mlp: true` (doubling the MLP width on KV-shared layers), whereas E4B sets it to `false`. Here `ple_layer` uses whatever width `params` provides.
- **Batching.** The library tensors include a leading batch axis.
- **`layer_scalar`.** In `transformers`, this is a buffer initialized to ones. Shipped checkpoint values were not inspected, and the tests use non-unit values.
- **Inputs without token IDs.** When called with embeddings only, `Gemma4TextModel` recovers token IDs by matching against the embedding matrix. Calling `project_per_layer_inputs` without the table term returns only the normalized projection without the $2^{-1/2}$ factor. This problem always provides `tokens`.
- **Pre-scaled `h0`.** In the model, `h0` comes from `embed_tokens(input_ids)` already multiplied by $\sqrt{D}$. Here `h0` is passed in directly.
- **KV-shared layers.** In the last `num_kv_shared_layers` layers, the model may keep only the token positions whose logits are needed and slice `per_layer_inputs` accordingly.
