Implement the forward pass of a decoder in the shape of the Qwen3.5 text models, where
three of every four layers are Gated DeltaNet layers and the fourth is a gated attention
layer, so that it can continue a sequence. The two kinds of layer carry two kinds of
state, a fixed-size one and a growing one, and one call has to thread both.

## Specification

```python
def layer_types(n_layers, interval): ...  # -> list of str
def hybrid_forward(params, cfg, tokens, state): ...  # -> (logits, state)
```

**`layer_types`** returns the kind of each layer as a list of `n_layers` strings: layer
$i$, counted from 0, is `"full_attention"` when $(i + 1) \bmod \texttt{interval} = 0$ and
`"linear_attention"` otherwise ($\texttt{n\_layers} \ge 1$, $\texttt{interval} \ge 1$).

**`hybrid_forward`** runs $T$ new tokens through the model. `cfg` is a
[`HybridConfig`](../../../python/src/learnai/kit/hybrid.py); write $D$ for `d_model`,
$H$ for `n_heads`, $G$ for `n_kv_heads`, and $d$ for `head_dim` (a field of its own:
$H d$ need not equal $D$). The kind of layer $i$ is `cfg.layers[i]`: the config's
explicit `layer_types` when it has one, and otherwise the interval rule above with
`cfg.full_attention_interval`. `params` maps the checkpoint's tensor names to float32
weights; linear weights are `(out, in)`, applied as `x @ W.T`, and there are no biases.
$P$ is the number of tokens the state has already seen (0 for a new sequence).

1. **Embed.** $x = E[\texttt{tokens}]$ with $E$ = `model.embed_tokens.weight`, shape
   $(T, D)$, not scaled.
2. **Blocks.** For each layer $i$ in order, with the weights `model.layers.{i}.*`:

$$
x \leftarrow x + \operatorname{Mix}_i\bigl(\operatorname{RMSNorm}(x;\, g_1)\bigr), \qquad
x \leftarrow x + \operatorname{MLP}_i\bigl(\operatorname{RMSNorm}(x;\, g_2)\bigr),
$$

   with $g_1$ = `input_layernorm.weight` and $g_2$ = `post_attention_layernorm.weight`.
   Every RMSNorm of the model, the two inside attention included, is
   `ops.rms_norm(x, weight, cfg.norm_eps)`: the weight is a plain gain. The MLP is
   `ops.swiglu(u, gate_proj, up_proj, down_proj)` with the three `mlp.*` weights.
3. **Head.** $\text{logits} = \operatorname{RMSNorm}(x;\, g)\,E^\top$ with $g$ =
   `model.norm.weight`. The head is tied to the embedding.

**A linear-attention mixer** is the layer of `gated-delta-layer`, from the kit:
`ops.gated_delta_layer(u, weights, layer_state, cfg.n_key_heads, cfg.norm_eps)` returns
`(out, (conv_state, S))`. `weights` maps the nine names of that problem
(`in_proj_qkv.weight`, `in_proj_z.weight`, `in_proj_a.weight`, `in_proj_b.weight`,
`conv1d.weight`, `A_log`, `dt_bias`, `norm.weight`, `out_proj.weight`) to the tensors
stored under `model.layers.{i}.linear_attn.`; `layer_state` is `None` or the pair an
earlier call returned.

**A full-attention mixer** is gated attention. For a normalized input $u$ of shape
$(T, D)$, with the weights under `model.layers.{i}.self_attn.`:

1. **Project.** $u\,W_q^\top$ has $2 d H$ columns. Head $h$ owns columns $2 d h$ to
   $2 d h + 2 d - 1$: the first $d$ of them are its query $q_h$, the last $d$ its gate
   $\gamma_h$. $u\,W_k^\top$ and $u\,W_v^\top$ have $d\,G$ columns; key-value head $j$
   owns columns $d j$ to $d j + d - 1$.
2. **Normalize** each query with `q_norm.weight` and each key with `k_norm.weight`, an
   RMSNorm over the $d$ channels of one head. Values and gates are not normalized.
3. **Rotate** queries and keys, not values, with
   `ops.partial_rope(x, positions, cfg.rope_base, cfg.partial_rotary_factor)`, the
   function of `partial-rope`: the first $\lfloor d \cdot \texttt{factor} \rfloor$
   channels of a head rotate and the rest pass through. Row $t$ of this call is the
   token at position $P + t$.
4. **Attend.** The keys and values of this call follow the cached ones, giving $P + T$
   per key-value head. The query at position $p$ attends to the keys at positions
   $0, \dots, p$; query head $h$ uses key-value head $\lfloor h / (H / G) \rfloor$; scores
   are divided by $\sqrt d$. This is `ops.attention(q, k, v, mask)` for `q` of shape
   $(H, T, d)$, `k` and `v` of shape $(G, P + T, d)$, and a boolean `mask` of shape
   $(T, P + T)$ that is `True` where a query may attend. Call the result $a$,
   $(H, T, d)$.
5. **Gate and project.** With heads laid side by side in head order,

$$
\text{out}_t = \bigl[\,a_{0,t} \odot \sigma(\gamma_{0,t}),\ \dots,\ a_{H-1,t} \odot \sigma(\gamma_{H-1,t})\,\bigr]\;W_o^\top ,
$$

   where $\sigma$ is the logistic sigmoid.

The layer's state is `(k, v)`: the keys **after the norm and the rotation** and the
values, each of shape $(G, P + T, d)$.

![The state of a four-layer model after four tokens. Each linear-attention layer keeps a pair of fixed size; the attention layer keeps a key and a value for every token, and the next token's position is the number of columns.](assets/two-kinds-of-state.svg)

Arguments and results:

- `tokens`: integer tensor of shape `(T,)`, $T \ge 1$, ids in `[0, cfg.vocab_size)`.
- `state`: `None` for a new sequence, or the list returned by the call that read the
  tokens before these. It has one entry per layer, in layer order: the pair
  `(conv_state, S)` of a linear-attention layer or the pair `(k, v)` of a
  full-attention layer. Nothing else is stored: $P$ is the length of the cached `k` of
  any full-attention layer. A model may have no such layer (an interval larger than
  `n_layers`); its state then holds only `(conv_state, S)` pairs, there is no $P$ to
  read, and none is needed, because only the attention mixer uses positions. A model
  may equally consist of attention layers only.
- Returns `(logits, state)`: float32 logits of shape `(T, cfg.vocab_size)`, where row
  $t$ scores the token after position $P + t$, and the state after these tokens as a
  new list.
- Nothing passed in is modified, neither a tensor nor the state list: a caller may
  continue twice from one state.

The kit's `ops` functions named above are the intended building blocks. Do not call
`hybrid_forward_ref` or `gated_attention_ref`.

**Stages.** The tests are grouped. Stage 1 calls only `layer_types`. Stage 2 calls
`hybrid_forward` with `state=None` and reads only the logits, so it passes before a
correct state is returned. Stage 3 checks the returned state and feeds it back.

## Shapes

For a linear-attention layer, as in `gated-delta-layer`: $K$ = `cfg.conv_kernel`,
$H_v$ = `cfg.n_value_heads`, $d_v$ = `cfg.value_head_dim`, $d_k$ = `cfg.key_head_dim`,
and $C = 2\,\texttt{n\_key\_heads}\,d_k + H_v d_v$ (`cfg.gdn.conv_dim`) is the width of
its convolution.

| name | shape | dtype |
|---|---|---|
| `tokens` | `(T,)` | integer |
| `model.embed_tokens.weight` | `(vocab_size, D)` | float32 |
| `input_layernorm.weight`, `post_attention_layernorm.weight`, `model.norm.weight` | `(D,)` | float32 |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | `(d_mlp, D)` | float32 |
| `mlp.down_proj.weight` | `(D, d_mlp)` | float32 |
| `self_attn.q_proj.weight` | `(H * 2 * d, D)` | float32 |
| `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | `(G * d, D)` | float32 |
| `self_attn.o_proj.weight` | `(D, H * d)` | float32 |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | `(d,)` | float32 |
| `linear_attn.*` | as in `gated-delta-layer` | float32 |
| state of a linear-attention layer: `conv_state`, `S` | `(K - 1, C)`, `(Hv, dv, dk)` | float32 |
| state of a full-attention layer: `k`, `v` | `(G, P + T, d)` each | float32 |
| `logits` | `(T, vocab_size)` | float32 |

## Example

```python
layer_types(4, 4)  # ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
layer_types(6, 2)  # linear, full, linear, full, linear, full
layer_types(3, 1)  # ["full_attention", "full_attention", "full_attention"]
```

A two-layer model (a linear-attention layer, then an attention layer) in the kit's
default sizes:

```python
# `tensor` stands for your framework's integer tensor, and `params` for its float32 weights.
cfg = HybridConfig(n_layers=2, full_attention_interval=2, rope_base=1e7)
params = hybrid_params(cfg, seed=3)

logits, state = hybrid_forward(params, cfg, tensor([3, 1, 4, 1, 5, 9]), None)
# logits[:, :4] = [[-1.386595, -0.069653, 4.239098,  0.426853],
#                  [-0.144362,  2.301395, 3.292907,  0.386430],
#                  [ 0.573805, -0.529514, 3.407029,  0.064500],
#                  [ 1.511575, -0.140076, 4.267532, -1.260582],
#                  [-1.357017, -0.347809, 2.133819, -3.233957],
#                  [ 0.675812, -1.043934, 0.745240, -0.798664]]
```

In two calls:

```python
_, state = hybrid_forward(params, cfg, tensor([3]), None)
# state[0] = (conv_state (3, 64), S (4, 8, 8));  state[1] = (k (2, 1, 16), v (2, 1, 16))
logits, state = hybrid_forward(params, cfg, tensor([1]), state)
# logits[:, :4] = row 1 above;  state[1] = (k (2, 2, 16), v (2, 2, 16))
```

In the second call the token `1` sits at position 1 and attends to keys 0 and 1. In
layer 0 the convolution reads two rows of zeros and the saved row of the token `3`
before its own.

## Constraints

- Every test uses the default sizes of `HybridConfig` (model width 32, 4 query heads on
  2 key-value heads of width 16, delta layers with 2 key heads and 4 value heads of
  width 8, kernel 4, vocabulary 64) and `tie_embeddings = True`. A sequence has at most
  8 tokens, a model at most 8 layers.
- Read `cfg.norm_eps` ($10^{-6}$ to $0.3$), `cfg.rope_base` ($50$ to $10^7$),
  `cfg.partial_rotary_factor` (0.25 or 0.5), and `cfg.layers` from the config.
- Logits and every state tensor are compared with relative tolerance `5e-4` and absolute
  tolerance `5e-4`, against the reference, the kit's `hybrid_forward_ref`, and the logits
  and cache of a `Qwen3_5ForCausalLM` in Hugging Face Transformers holding the same
  weights. The library sums in another order and runs the delta rule in chunks; the
  largest difference measured between it and a correct solution is below `4e-5`.
- Any split of a sequence into consecutive calls, each given the state the previous one
  returned, gives the logits and the final state of a single call, to the same
  tolerance. The splits of 8 tokens that are tested are 6 + 1 + 1 and 1 + 1 + 1 + 5 (a
  first call of one token, and calls shorter than the $K - 1 = 3$ rows of convolution
  state), and a continuation of 3 tokens by 5.

## Why this matters

A serving stack written for attention assumes that state is a cache: it grows by one
entry per token, can be truncated to roll back, and can be rebuilt by running the prompt
again. In Qwen3.5-0.8B, 18 of 24 layers break each of those assumptions, and the 6
attention layers still need all of them. A decode loop that treats the per-layer state
as one opaque kind either restarts the recurrent layers at every call or forgets where
in the sequence the attention layers are; both produce fluent, wrong text, and both
pass every test that calls the model once.
