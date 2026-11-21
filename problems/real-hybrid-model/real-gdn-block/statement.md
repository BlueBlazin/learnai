Implement one linear-attention block of a real model, `Qwen/Qwen3.5-0.8B`, from the
tensors in its checkpoint, and match the hidden states that `transformers` computes.
The mixer is the Gated DeltaNet layer of `gated-delta-layer`; what makes it this
model's block is how the checkpoint stores its norm weights, which differs between two
norms of the same block, and an epsilon that only hidden states of the real size reveal.

## Specification

```python
def qwen_gdn_block(x, weights, layer, state): ...  # -> (x, (conv_state, S))
```

- `x`: the hidden states of $T \ge 1$ consecutive tokens entering the block, float32,
  shape `(T, D)`.
- `weights`: a mapping from checkpoint names to float32 tensors (a dict, or the kit's
  lazy [`Weights`](../../../python/src/learnai/kit/realmodel.py), which is indexed and
  iterated the same way). It holds the whole model. With
  `p = "model.language_model.layers.{layer}."`, the block reads the 14 tensors in the
  table under [Shapes](#shapes). Matrices are `(out, in)`, applied as `u @ W.T`; there
  are no biases.
- `layer`: the index of the block, an int. It is a linear-attention layer.
- `state`: `None` when `x` starts a sequence, or the pair `(conv_state, S)` returned by
  the call that read the tokens before `x`.
- Returns `(x, (conv_state, S))`: the hidden states leaving the block, float32
  `(T, D)`, and the state after the last token. The tensors passed in are not modified.

The sizes are not arguments; read them from the shapes. $H$ is the number of heads
(the length of `A_log`; this model has as many key heads as value heads), $d_v$ the
length of `linear_attn.norm.weight`, $d_k$ follows from the rows of `in_proj_qkv`,
$K$ is the kernel length, and $C = 2 H d_k + H d_v$. In Qwen3.5 0.8B: $D = 1024$,
$H = 16$, $d_k = d_v = 128$, $K = 4$, $C = 6144$; the tests also use a smaller model
with $d_k \ne d_v$.

$$
x \leftarrow x + \operatorname{Mixer}\big(\operatorname{Norm}(x;\, w_1)\big), \qquad
x \leftarrow x + \operatorname{MLP}\big(\operatorname{Norm}(x;\, w_2)\big),
$$

with $w_1$ = `p + "input_layernorm.weight"` and $w_2$ =
`p + "post_attention_layernorm.weight"`.

**The block norms store their gain minus one.**

$$
\operatorname{Norm}(u; w) = \frac{u}{\sqrt{\operatorname{mean}(u^2) + 10^{-6}}} \odot (1 + w),
$$

the mean over the last axis: `ops.rms_norm(u, 1 + w, 1e-6)`. The epsilon is
$10^{-6}$; `ops.rms_norm` defaults to another value.

**Mixer**, for the normalized rows $u$, with the tensors under `p + "linear_attn."`:

1. $P = u\,W_{qkv}^\top$, shape $(T, C)$. Let $\tilde P$ be the $K - 1$ rows of
   `conv_state` (zeros if `state` is `None`) followed by the rows of $P$. With $w$ =
   `conv1d.weight`, the last tap on the current row:
   $M_{t,c} = \operatorname{silu}\big(\sum_{j=0}^{K-1} w_{c,0,j}\,\tilde P_{t+j,\,c}\big)$.
   The new `conv_state` is the last $K - 1$ rows of $\tilde P$. This is
   `ops.causal_conv1d(P, w, conv_state)` followed by SiLU.
2. The first $H d_k$ columns of $M$ are the queries, `(T, H, dk)`; the next $H d_k$
   the keys; the last $H d_v$ the values, `(T, H, dv)`. Queries and keys are normalized
   over the last axis with $\operatorname{l2norm}(a) = a / \sqrt{\sum_i a_i^2 + 10^{-6}}$,
   and the queries are then multiplied by $d_k^{-1/2}$.
3. Gates per token and head, with $a = u\,W_a^\top$ and $b = u\,W_b^\top$:
   $g = \exp\big(-e^{\texttt{A\_log}} \cdot \operatorname{softplus}(a + \texttt{dt\_bias})\big)$
   and $\beta = \sigma(b)$.
4. For each head, starting from its matrix in `S` (zeros if `state` is `None`) and for
   $t = 0, \dots, T - 1$ in order: $\bar S = g_t S$, then
   $S = \bar S + \beta_t\,(v_t - \bar S k_t)\,k_t^\top$, then $y_t = S\,q_t$.
5. **The gated norm stores a plain gain.** With $z = u\,W_z^\top$ reshaped to
   `(T, H, dv)` and $\gamma$ = `norm.weight`, used as stored:
   $o = \gamma \odot y \,/ \sqrt{\operatorname{mean}(y^2) + 10^{-6}} \odot \operatorname{silu}(z)$
   per token and head (norm first, then the gate), and the mixer's output is
   $\operatorname{flatten}(o)\,W_{out}^\top$.

**MLP**: SwiGLU, $\operatorname{MLP}(u) = \big(\operatorname{silu}(u\,W_{gate}^\top) \odot u\,W_{up}^\top\big)\,W_{down}^\top$,
which is `ops.swiglu(u, W_gate, W_up, W_down)`.

Compute the block yourself. From the kit, `ops.rms_norm`, `ops.swiglu`, `ops.l2norm`,
`ops.softplus`, and `ops.causal_conv1d` are meant to be used. Not allowed:
`ops.gated_delta_layer` (under any import path), the kit's delta-rule oracles and
reference blocks and models (`gdn_step_ref`, `gdn_gates_ref`, `gdn_layer_ref`,
`hybrid_block_ref`, `hybrid_forward_ref`, `RealHybrid`, `RealHybridLM`), and the
packages `transformers`, `safetensors`, and `mlx_lm`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x`, result `x` | `(T, D)` | float32 |
| `conv_state` | `(K - 1, C)` | float32 |
| `S` | `(H, dv, dk)` | float32 |
| `p + "input_layernorm.weight"`, `p + "post_attention_layernorm.weight"` | `(D,)` | float32 |
| `p + "linear_attn.in_proj_qkv.weight"` | `(C, D)` | float32 |
| `p + "linear_attn.conv1d.weight"` | `(C, 1, K)` | float32 |
| `p + "linear_attn.in_proj_a.weight"`, `in_proj_b.weight` | `(H, D)` | float32 |
| `p + "linear_attn.A_log"`, `dt_bias` | `(H,)` | float32 |
| `p + "linear_attn.in_proj_z.weight"` | `(H * dv, D)` | float32 |
| `p + "linear_attn.norm.weight"` | `(dv,)` | float32 |
| `p + "linear_attn.out_proj.weight"` | `(D, H * dv)` | float32 |
| `p + "mlp.gate_proj.weight"`, `up_proj.weight` | `(F, D)` | float32 |
| `p + "mlp.down_proj.weight"` | `(D, F)` | float32 |

In Qwen3.5 0.8B: `in_proj_qkv` is `(6144, 1024)`, `conv1d` `(6144, 1, 4)`, `in_proj_z`
`(2048, 1024)`, `out_proj` `(1024, 2048)`, and $F = 3584$; `conv_state` is `(3, 6144)`
and `S` is `(16, 128, 128)`.

## Example

The first public case runs layer 1 of a four-layer model with random weights in the
checkpoint's layout: $D = 32$, $H = 4$, $d_k = 4$, $d_v = 8$, $K = 4$ (so $C = 64$),
$F = 48$. Its stored `input_layernorm.weight` has mean 0.074, a gain of 1.074. The
input is six rows of standard normal numbers.

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

Row 5 depends on the rows before it twice: through the three rows of `conv_state` and
through the memory `S`. With `state=None` both are zeros and the row comes out as if it
were the second token of a new sequence.

## Constraints

- Hidden states are compared to a relative tolerance of `1e-3` and an absolute
  tolerance of `1e-5`, the returned `conv_state` and `S` to `1e-3` and `1e-4`. Results
  must be float32.
- $1 \le T \le 12$. A call may have fewer than $K - 1$ rows, and a call of one row
  (a decode step) is the common case in a server. Any split of a sequence into
  consecutive calls, each given the state the previous one returned, must give the
  outputs of one call, to the same tolerance. A state passed in may come from the kit
  and not from your own function, so the layout of `conv_state` and `S` matters.
- `x`, `conv_state`, and `S` must hold the same values after the call as before it: a
  caller may continue twice from one state. The tests check this on tensors they keep.
- The cases without a download use the random model of the example; expected values
  come from the kit's numpy block on the same weights. One of them scales the input by
  0.03, the size of the real model's hidden states, where an epsilon of $10^{-5}$ in
  place of $10^{-6}$ changes a norm by about half a percent.
- The other cases run layers 1 and 13 of `Qwen/Qwen3.5-0.8B` (revision `2fc0636`) on
  the hidden states of a 12-token prompt, in one call and as 5 + 7 rows. Inputs and
  expected outputs are hidden states computed by `transformers` 5.23 in float32 in one
  call, compared at 64 fixed columns of the 1024. A typical entry is 0.02 to 0.04.
- In the real model, read only the tensors of the block you were asked for: `weights`
  decodes a tensor from bfloat16 the first time it is indexed, and layer 1's names
  are a prefix of the names of layers 10 to 19.

## Why this matters

Nothing in a checkpoint says whether a norm weight is a gain or an offset from one.
Qwen3.5 stores the block norms, the final norm, and the attention layers' query and
key norms as offsets, and the norm inside each Gated DeltaNet layer as a gain; its
reference implementation adds the one at run time for some and not for others. A port
that treats all of them alike loads without a complaint and runs: with offsets used as
gains, the normalized stream is multiplied by numbers around 0.4 and 0.2 instead of 1.4
and 1.2, and the text is merely worse. Comparing one block at a time against hidden
states from the reference implementation is how such a mistake is found, and the
block's returned state is what a serving engine has to save per sequence and per layer
instead of a key-value cache.
