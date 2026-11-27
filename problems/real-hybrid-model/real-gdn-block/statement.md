Implement one linear-attention block of `Qwen/Qwen3.5-0.8B` directly from its checkpoint tensors so that its hidden states match `transformers`. The block combines the Gated DeltaNet mixer from `gated-delta-layer` with a SwiGLU MLP, using Qwen3.5's checkpoint norm conventions and RMSNorm epsilon.

## Specification

```python
def qwen_gdn_block(x, weights, layer, state): ...  # -> (x, (conv_state, S))
```

- `x`: float32 hidden states for $T \ge 1$ consecutive tokens of hidden width $D$, with shape `(T, D)`.
- `weights`: a mapping from checkpoint tensor names to float32 tensors (either a dict or the kit's lazy [`Weights`](../../../python/src/learnai/kit/realmodel.py) loader, which supports the same indexing and iteration). It holds the full model. With `p = f"model.language_model.layers.{layer}."`, this block reads the 14 tensors listed in [Shapes](#shapes). Linear projection matrices have shape `(out, in)`, are applied as `u @ W.T`, and have no biases.
- `layer`: integer index of a linear-attention layer.
- `state`: `None` at the start of a sequence, or the `(conv_state, S)` pair returned by the previous call on earlier tokens.
- Returns `(x, (conv_state, S))`: the float32 hidden states leaving the block and the updated recurrent state after the last token. Do not modify any input tensor in place.

Infer all layer dimensions from tensor shapes rather than hardcoding them. Let $H$ be the number of heads (the length of `linear_attn.A_log`; key heads and value heads have the same count), $d_v$ the length of `linear_attn.norm.weight`, $K$ the convolution kernel length (the last axis of `linear_attn.conv1d.weight`), and $F$ the MLP intermediate width (the row count of `mlp.gate_proj.weight`). The total projected QKV width is $C = 2 H d_k + H d_v$, which determines $d_k$ from the row count of `linear_attn.in_proj_qkv.weight`. In Qwen3.5 0.8B, $D = 1024$, $H = 16$, $d_k = d_v = 128$, $K = 4$, $C = 6144$, and $F = 3584$. Synthetic tests also use a smaller model where $d_k \ne d_v$.

Each block applies a pre-norm residual mixer followed by a pre-norm residual MLP:

$$
x \leftarrow x + \operatorname{Mixer}\big(\operatorname{Norm}(x;\, w_1)\big), \qquad
x \leftarrow x + \operatorname{MLP}\big(\operatorname{Norm}(x;\, w_2)\big),
$$

where $w_1$ is `p + "input_layernorm.weight"` and $w_2$ is `p + "post_attention_layernorm.weight"`.

### Block norms (offset convention)

The two block norms store their gain minus one, so add `1` before scaling:

$$
\operatorname{Norm}(u; w) = \frac{u}{\sqrt{\operatorname{mean}(u^2) + 10^{-6}}} \odot (1 + w),
$$

with the mean taken over the last axis. This matches `ops.rms_norm(u, 1 + w, 1e-6)`. Pass `1e-6` explicitly because `ops.rms_norm` uses a different default epsilon.

### Mixer

Given normalized hidden states $u$ of shape `(T, D)`, read the mixer tensors under `p + "linear_attn."`:

1. **QKV projection and causal convolution.** Project $P = u\,W_{qkv}^\top$ using `in_proj_qkv.weight`, giving shape `(T, C)`. Form $\tilde P$ by prepending the $K - 1$ rows of `conv_state` (zeros if `state` is `None`) to the rows of $P$. Convolve each channel $c$ with `w = conv1d.weight`, placing the last tap on the current token, and apply SiLU:

$$
M_{t,c} = \operatorname{silu}\!\left(\sum_{j=0}^{K-1} w_{c,0,j}\,\tilde P_{t+j,\,c}\right).
$$

   The new `conv_state` is the last $K - 1$ rows of $\tilde P$. This is `ops.causal_conv1d(P, w, conv_state)` followed by SiLU.
2. **Split and normalize Q, K, V.** Slice $M$ along the channel axis into queries (first $H d_k$ columns, reshaped to `(T, H, dk)`), keys (next $H d_k$ columns, `(T, H, dk)`), and values (last $H d_v$ columns, `(T, H, dv)`). L2-normalize queries and keys along the last axis with $\operatorname{l2norm}(a) = a / \sqrt{\sum_i a_i^2 + 10^{-6}}$, then scale queries by $d_k^{-1/2}$.
3. **Gates.** Project $a = u\,W_a^\top$ (`in_proj_a.weight`) and $b = u\,W_b^\top$ (`in_proj_b.weight`), both of shape `(T, H)`. Compute per-token, per-head decay and update gates:

$$
g = \exp\!\big(-e^{\texttt{A\_log}} \cdot \operatorname{softplus}(a + \texttt{dt\_bias})\big), \qquad \beta = \sigma(b),
$$

   where $\sigma$ is the sigmoid function.
4. **Delta-rule recurrence.** Start from `S` (zeros of shape `(H, dv, dk)` if `state` is `None`). For each head and each step $t = 0, \dots, T - 1$ in order, let $q_t, k_t, v_t, g_t, \beta_t$ be that head's slices at step $t$ and update:

$$
\bar S = g_t S, \qquad S = \bar S + \beta_t\,(v_t - \bar S k_t)\,k_t^\top, \qquad y_t = S\,q_t.
$$

5. **Gated norm (plain gain) and output projection.** Project $z = u\,W_z^\top$ (`in_proj_z.weight`) and reshape to `(T, H, dv)`. Unlike the block norms, $\gamma$ = `norm.weight` is stored as a plain gain and used without adding `1`. Normalize $y$ along the last axis first, then multiply by the gate:

$$
o = \gamma \odot \frac{y}{\sqrt{\operatorname{mean}(y^2) + 10^{-6}}} \odot \operatorname{silu}(z).
$$

   Flatten $o$ to `(T, H * dv)` and project with `out_proj.weight` (`W_out`) to get $\operatorname{flatten}(o)\,W_{out}^\top$.

### MLP

Compute a SwiGLU feed-forward layer using `mlp.gate_proj.weight` ($W_{gate}$), `mlp.up_proj.weight` ($W_{up}$), and `mlp.down_proj.weight` ($W_{down}$):

$$
\operatorname{MLP}(u) = \big(\operatorname{silu}(u\,W_{gate}^\top) \odot u\,W_{up}^\top\big)\,W_{down}^\top,
$$

which is `ops.swiglu(u, W_gate, W_up, W_down)`.

### Allowed and forbidden helpers

Compute the block yourself. You may use `ops.rms_norm`, `ops.swiglu`, `ops.l2norm`, `ops.softplus`, and `ops.causal_conv1d` from the kit. Do not call `ops.gated_delta_layer` (under any import path) or any of the kit's delta-rule oracles and reference blocks/models (`gdn_step_ref`, `gdn_gates_ref`, `gdn_layer_ref`, `hybrid_block_ref`, `hybrid_forward_ref`, `RealHybrid`, `RealHybridLM`). Do not import `transformers`, `safetensors`, or `mlx_lm`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x`, returned `x` | `(T, D)` | float32 |
| `conv_state` | `(K - 1, C)` | float32 |
| `S` | `(H, dv, dk)` | float32 |
| `p + "input_layernorm.weight"`, `p + "post_attention_layernorm.weight"` | `(D,)` | float32 |
| `p + "linear_attn.in_proj_qkv.weight"` | `(C, D)` | float32 |
| `p + "linear_attn.conv1d.weight"` | `(C, 1, K)` | float32 |
| `p + "linear_attn.in_proj_a.weight"`, `p + "linear_attn.in_proj_b.weight"` | `(H, D)` | float32 |
| `p + "linear_attn.A_log"`, `p + "linear_attn.dt_bias"` | `(H,)` | float32 |
| `p + "linear_attn.in_proj_z.weight"` | `(H * dv, D)` | float32 |
| `p + "linear_attn.norm.weight"` | `(dv,)` | float32 |
| `p + "linear_attn.out_proj.weight"` | `(D, H * dv)` | float32 |
| `p + "mlp.gate_proj.weight"`, `p + "mlp.up_proj.weight"` | `(F, D)` | float32 |
| `p + "mlp.down_proj.weight"` | `(D, F)` | float32 |

In Qwen3.5 0.8B, `in_proj_qkv` is `(6144, 1024)`, `conv1d` is `(6144, 1, 4)`, `in_proj_z` is `(2048, 1024)`, `out_proj` is `(1024, 2048)`, `conv_state` is `(3, 6144)`, and `S` is `(16, 128, 128)`.

## Example

The first public test runs layer 1 of a 4-layer random model with the checkpoint's naming layout: $D = 32$, $H = 4$, $d_k = 4$, $d_v = 8$, $K = 4$ ($C = 64$), and $F = 48$. Its stored `input_layernorm.weight` has mean `0.074`, giving an effective gain of `1.074`. The input `x` has six rows drawn from a standard normal distribution.

```python
y, (conv_state, S) = qwen_gdn_block(x, weights, 1, None)  # x: (6, 32)
x[5, :4]  # [ 0.899566, -0.236663, -0.629355,  0.231511]
y[5, :4]  # [ 0.775550, -0.210093, -0.297908,  0.922919]
conv_state.shape, S.shape  # (3, 64), (4, 8, 4)

# The same six rows in two calls:
y_a, state = qwen_gdn_block(x[:4], weights, 1, None)
y_b, _ = qwen_gdn_block(x[4:], weights, 1, state)
y_b[1, :4]  # [ 0.775550, -0.210093, -0.297908,  0.922919]

# And what the last two rows give without the state:
y_c, _ = qwen_gdn_block(x[4:], weights, 1, None)
y_c[1, :4]  # [ 2.594700,  1.175721,  1.260947,  1.556884]
```

Row 5 depends on earlier rows through both the three rows of `conv_state` and the recurrent memory `S`. Passing `state=None` resets both to zeros, so row 5 is processed as the second token of a new sequence.

## Constraints

- Hidden states must match within relative tolerance `1e-3` and absolute tolerance `1e-5`. Returned `conv_state` and `S` must match within `1e-3` relative and `1e-4` absolute. All outputs must be float32.
- $1 \le T \le 12$. A call may have fewer than $K - 1$ rows, including $T = 1$ (a single decode step). Splitting any sequence across consecutive calls with carried state must match a single call over the full sequence within the same tolerances. Input states may come from the kit rather than your own function, so the exact layouts of `conv_state` and `S` matter.
- `x`, `conv_state`, and `S` must remain unchanged after the call because a caller may branch multiple continuations from the same state. Tests verify this directly.
- Non-download tests use the small random model from the example, checked against the kit's NumPy reference block. One test scales the input by `0.03` to match the magnitude of real hidden states, where using epsilon `1e-5` instead of `1e-6` shifts a norm by roughly 0.5%.
- Real-model tests run layers 1 and 13 of `Qwen/Qwen3.5-0.8B` (revision `2fc0636`) on a 12-token prompt, both as one call of 12 rows and as 5 + 7 rows. Expected outputs are float32 hidden states from `transformers` 5.23, compared at 64 fixed columns of the 1024. Typical entries are `0.02` to `0.04`.
- On the real model, read only tensors belonging to the requested layer. `weights` lazily decodes tensors from bfloat16 on first access, and layer 1's prefix without a trailing dot matches layers 10 through 19.

## Why this matters

A checkpoint file does not record whether a norm weight is a plain gain or an offset from one. Qwen3.5 stores its block norms, final norm, and attention query/key norms as offsets, while storing the norm inside each Gated DeltaNet layer as a plain gain. The reference code adds `1` at runtime for the offset norms and leaves the gated norm untouched.

A port that treats every norm weight the same way loads without errors and still produces text. Using offset weights as plain gains scales the residual stream by roughly `0.4` and `0.2` instead of `1.4` and `1.2`, degrading quality silently. Checking one block at a time against reference hidden states catches this mismatch immediately, and the `(conv_state, S)` pair returned by the block is the fixed-size state a serving engine must store per sequence and per layer in place of a KV cache.
