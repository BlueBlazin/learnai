Gemma 4's E2B and E4B models look every token up in a second embedding table whose row
is a pack of one small vector per decoder layer, and each layer adds its own vector to
the residual stream as its last step. Implement the lookup, the injection, and one
layer that uses them.

![This problem is the per-layer embedding, which each block adds to the stream as its last step.](assets/map.svg)

## Specification

```python
def ple_inputs(tokens, h0, table, w_model, norm_gain, n_layers): ...  # -> (T, L, d_ple)
def ple_inject(h, ple_i, w_gate, w_proj, post_gain): ...  # -> (T, D)
def ple_layer(params, x, ple, i, mask): ...  # -> (T, D)
```

Throughout, $\operatorname{rms}(x, g) = g \odot x \,/ \sqrt{\operatorname{mean}(x^2) + 10^{-6}}$
with the mean over the last axis: `ops.rms_norm(x, g, eps=1e-6)` from `learnai.kit`
(the epsilon is not the kit's default). Linear maps have weights of shape `(out, in)`
and no bias. $D$ is the stream width, $L$ = `n_layers`, and $d_\text{ple}$ is the
width of `table` divided by $L$.

**`ple_inputs`** returns what every layer will receive, for all $T$ tokens. It has two
parts of shape `(T, L * d_ple)`:

$$
e = \sqrt{d_\text{ple}}\;\texttt{table}[\texttt{tokens}], \qquad
c = D^{-1/2}\; h_0\, W_\text{model}^\top .
$$

Each row of both is cut into $L$ consecutive slices of width $d_\text{ple}$: slice $i$
is columns $i\,d_\text{ple}$ to $(i + 1)\,d_\text{ple} - 1$ and belongs to layer $i$.
With $e_{t,i}$ and $c_{t,i}$ the slices of token $t$,

$$
\texttt{ple}[t, i] = 2^{-1/2}\,\big(\operatorname{rms}(c_{t,i},\ \texttt{norm\_gain}) + e_{t,i}\big),
$$

so the norm runs over the $d_\text{ple}$ channels of one slice, after the scaling by
$D^{-1/2}$. `h0` is the input of layer 0 (the scaled token embeddings), used as given.

**`ple_inject`** adds one layer's slice `ple_i`, shape `(T, d_ple)`, to a stream `h`:

$$
\begin{aligned}
\texttt{ple\_inject}(h, p) &= h + \operatorname{rms}\Big(\big(\operatorname{gelu}(h\,W_\text{gate}^\top) \odot p\big)\, W_\text{proj}^\top,\ \texttt{post\_gain}\Big), \\
\operatorname{gelu}(z) &= \tfrac{1}{2}\, z \left(1 + \tanh\!\big(\sqrt{2/\pi}\,(z + 0.044715\, z^3)\big)\right).
\end{aligned}
$$

(the tanh form of GELU). The norm is on the branch only.

**`ple_layer`** is layer `i` of the decoder, with the kit's single-head attention
standing in for Gemma 4's. Each sublayer reads a normalized copy of the stream and has
its output normalized again; the per-layer embedding enters after both:

$$
\begin{aligned}
a &= x + \operatorname{rms}\big(\operatorname{Attn}(\operatorname{rms}(x, g_1)),\ g_2\big), \\
m &= a + \operatorname{rms}\big(\operatorname{MLP}(\operatorname{rms}(a, g_3)),\ g_4\big), \\
y &= s \cdot \texttt{ple\_inject}(m,\ \texttt{ple}[:, i]) .
\end{aligned}
$$

- $\operatorname{Attn}(u) = \texttt{ops.attention}(u W_q^\top,\ u W_k^\top,\ u W_v^\top,\ \texttt{mask})\; W_o^\top$:
  one head, no rotary positions. `mask` is boolean, `True` where the query (row) may
  attend to the key (column); every row has at least one `True`.
- $\operatorname{MLP}(u) = \big(\operatorname{gelu}(u W_\text{gate}^\top) \odot (u W_\text{up}^\top)\big)\, W_\text{down}^\top$,
  with the same GELU.
- `ple` is the whole result of `ple_inputs`; `i` is a Python int, $0 \le i < L$; $s$
  is `params["layer_scalar"]` and multiplies the final output, injection included.
- No function modifies a tensor it is given.

`params` holds the layer's weights under the names of the checkpoint:

| symbol | key | shape |
|---|---|---|
| $g_1$, $g_2$ | `input_layernorm.weight`, `post_attention_layernorm.weight` | `(D,)` |
| $W_q, W_k, W_v, W_o$ | `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, `self_attn.o_proj.weight` | `(D, D)` |
| $g_3$, $g_4$ | `pre_feedforward_layernorm.weight`, `post_feedforward_layernorm.weight` | `(D,)` |
| $W_\text{gate}, W_\text{up}$; $W_\text{down}$ of the MLP | `mlp.gate_proj.weight`, `mlp.up_proj.weight`; `mlp.down_proj.weight` | `(M, D)`; `(D, M)` |
| `w_gate`, `w_proj` of the injection | `per_layer_input_gate.weight`, `per_layer_projection.weight` | `(d_ple, D)`, `(D, d_ple)` |
| `post_gain`; $s$ | `post_per_layer_input_norm.weight`; `layer_scalar` | `(D,)`; `(1,)` |

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens`; `mask` | `(T,)`; `(T, T)` | integer; bool |
| `h0`, `h`, `x`, results of `ple_inject` and `ple_layer` | `(T, D)` | float32 |
| `table`; `w_model`; `norm_gain` | `(V, L * d_ple)`; `(L * d_ple, D)`; `(d_ple,)` | float32 |
| `ple`, result of `ple_inputs`; `ple_i` | `(T, L, d_ple)`; `(T, d_ple)` | float32 |

## Example

One token, $D = 2$, two layers with $d_\text{ple} = 2$, all gains 1:

```python
tokens = [1]
h0 = [[2.0, 0.0]]
table = [[9, 9, 9, 9], [1, 0, 0, 2], [9, 9, 9, 9]]
w_model = [[1, 0], [1, 0], [0, 1], [3, 0]]

ple = ple_inputs(tokens, h0, table, w_model, [1.0, 1.0], 2)
# [[[1.707106, 0.707106], [0.0, 3.0]]]

ple_inject([[1.0, -1.0]], ple[:, 0], eye(2), eye(2), [1.0, 1.0])
# [[2.409909, -1.110254]]
```

The table row of token 1 times $\sqrt 2$ is $(1.4142, 0 \mid 0, 2.8284)$, one slice per
layer. The projection $h_0 W^\top = (2, 2, 0, 6)$ times $2^{-1/2}$ is
$(1.4142, 1.4142 \mid 0, 4.2426)$; layer 0's slice has mean square 2 and normalizes to
$(1, 1)$, layer 1's has mean square 9 and normalizes to $(0, 1.4142)$. The sums
$(2.4142, 1)$ and $(0, 4.2426)$ times $2^{-1/2}$ are the result.

## Constraints

- $1 \le T \le 64$, $4 \le D \le 64$, $1 \le L \le 35$, $2 \le d_\text{ple} \le 16$,
  $V \le 64$; token ids are in $[0, V)$ and may repeat.
- Entries of `h0`, `h`, and `x` have magnitudes from about $10^{-3}$ to 10.
- Results are compared with the reference and, on a tiny configuration, with arrays
  computed by `Gemma4TextModel` and `Gemma4TextDecoderLayer` of Hugging Face
  Transformers: relative tolerance `1e-4`, absolute tolerance `1e-5`.
- Layer $i$ of the result of `ple_inputs` depends on no column of `table` and no row
  of `w_model` outside slice $i$.

## Why this matters

The E2B configuration has 35 layers, $d_\text{ple} = 256$, and a vocabulary of 262144,
so the second table is 8960 wide and holds 2.35 billion numbers, against 0.40 billion
in the main embedding of width 1536, and costs one row read per token instead of a
matrix product. A loader that cuts the packed row in the wrong order, hands every layer
the first slice, or injects before the MLP still produces logits from the main path,
with every layer's token-specific signal scrambled and no error raised.
