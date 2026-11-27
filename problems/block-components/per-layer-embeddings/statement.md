Gemma 4's E2B and E4B models look up every token in a second embedding table that packs one small vector per decoder layer, and each layer injects its own vector into the residual stream as its final step. Implement the per-layer lookup, the gated injection, and one decoder layer that uses them.

![This problem is the per-layer embedding, which each block adds to the stream as its last step.](assets/map.svg)

## Specification

```python
def ple_inputs(tokens, h0, table, w_model, norm_gain, n_layers): ...  # -> (T, L, d_ple)
def ple_inject(h, ple_i, w_gate, w_proj, post_gain): ...  # -> (T, D)
def ple_layer(params, x, ple, i, mask): ...  # -> (T, D)
```

Let $T$ be the sequence length, $D$ the residual stream width, $L$ the number of layers (`n_layers`), $V$ the vocabulary size, and $d_{\text{ple}}$ the per-layer embedding width (`table.shape[1] // L`). All linear weights use `(out, in)` layout with no bias and are applied to row vectors as `x @ w.T`.

Throughout this problem, $\operatorname{rms}(x, g)$ denotes RMS normalization over the last axis with $\epsilon = 10^{-6}$:

$$
\operatorname{rms}(x, g) = g \odot \frac{x}{\sqrt{\operatorname{mean}(x^2) + 10^{-6}}} ,
$$

where $\odot$ is elementwise multiplication. This is `ops.rms_norm(x, g, eps=1e-6)` from `learnai.kit` (note that `1e-6` is not the kit's default).

### `ple_inputs`

Computes the per-layer inputs for all $T$ tokens and all $L$ layers. It combines a token-table lookup $e$ and a projection $c$ of the layer-0 input `h0` (the already-scaled token embeddings, used as given), both of shape `(T, L * d_ple)`:

$$
e = \sqrt{d_{\text{ple}}}\;\texttt{table}[\texttt{tokens}], \qquad
c = D^{-1/2}\; h_0\, W_{\text{model}}^\top .
$$

Split each row of $e$ and $c$ into $L$ consecutive slices of width $d_{\text{ple}}$, so slice $i$ covers columns $i\,d_{\text{ple}}$ through $(i + 1)\,d_{\text{ple}} - 1$ and belongs to layer $i$. Writing $e_{t,i}$ and $c_{t,i}$ for layer $i$'s slices of token $t$, combine them as:

$$
\texttt{ple}[t, i] = 2^{-1/2}\,\big(\operatorname{rms}(c_{t,i},\ \texttt{norm\_gain}) + e_{t,i}\big) .
$$

The RMS norm runs over the $d_{\text{ple}}$ channels of each slice of $c$ after scaling by $D^{-1/2}$.

### `ple_inject`

Injects layer $i$'s slice `ple_i` ($p$) into the residual stream `h`. It projects `h` down to width $d_{\text{ple}}$, applies the tanh approximation of GELU, gates `ple_i` elementwise, projects back to width $D$, normalizes the injected branch only, and adds it to `h`:

$$
\begin{aligned}
\texttt{ple\_inject}(h, p) &= h + \operatorname{rms}\Big(\big(\operatorname{gelu}(h\,W_{\text{gate}}^\top) \odot p\big)\, W_{\text{proj}}^\top,\ \texttt{post\_gain}\Big), \\
\operatorname{gelu}(z) &= \tfrac{1}{2}\, z \left(1 + \tanh\!\big(\sqrt{2/\pi}\,(z + 0.044715\, z^3)\big)\right) .
\end{aligned}
$$

### `ple_layer`

Computes decoder layer `i` (a Python `int` with $0 \le i < L$) using single-head attention from `learnai.kit` in place of Gemma 4's attention. Each sublayer reads a normalized copy of the stream and normalizes its output before the residual add. The per-layer embedding is injected after the MLP, and `layer_scalar` ($s$) scales the entire final output:

$$
\begin{aligned}
a &= x + \operatorname{rms}\big(\operatorname{Attn}(\operatorname{rms}(x, g_1)),\ g_2\big), \\
m &= a + \operatorname{rms}\big(\operatorname{MLP}(\operatorname{rms}(a, g_3)),\ g_4\big), \\
y &= s \cdot \texttt{ple\_inject}(m,\ \texttt{ple}[:, i]) .
\end{aligned}
$$

- $\operatorname{Attn}(u) = \texttt{ops.attention}(u W_q^\top,\ u W_k^\top,\ u W_v^\top,\ \texttt{mask})\; W_o^\top$ is single-head attention with no rotary positions. `mask` is boolean and `True` where query row $q$ may attend to key column $k$, with at least one `True` per row.
- $\operatorname{MLP}(u) = \big(\operatorname{gelu}(u W_{\text{gate}}^\top) \odot (u W_{\text{up}}^\top)\big)\, W_{\text{down}}^\top$, using the same tanh-GELU and MLP inner width $M$.
- `ple` is the full output of `ple_inputs`.
- None of the three functions may modify any input tensor in place.

`params` is a dictionary holding the layer's weights under their checkpoint names:

| symbol | key | shape |
|---|---|---|
| $g_1$ | `input_layernorm.weight` | `(D,)` |
| $g_2$ | `post_attention_layernorm.weight` | `(D,)` |
| $W_q, W_k, W_v, W_o$ | `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, `self_attn.o_proj.weight` | `(D, D)` |
| $g_3$ | `pre_feedforward_layernorm.weight` | `(D,)` |
| $g_4$ | `post_feedforward_layernorm.weight` | `(D,)` |
| MLP $W_{\text{gate}}, W_{\text{up}}$ | `mlp.gate_proj.weight`, `mlp.up_proj.weight` | `(M, D)` |
| MLP $W_{\text{down}}$ | `mlp.down_proj.weight` | `(D, M)` |
| injection `w_gate` | `per_layer_input_gate.weight` | `(d_ple, D)` |
| injection `w_proj` | `per_layer_projection.weight` | `(D, d_ple)` |
| `post_gain` | `post_per_layer_input_norm.weight` | `(D,)` |
| $s$ | `layer_scalar` | `(1,)` |

## Shapes

| name | shape | dtype |
|---|---|---|
| `tokens` | `(T,)` | integer |
| `mask` | `(T, T)` | bool |
| `h0`, `h`, `x` | `(T, D)` | float32 |
| `table` | `(V, L * d_ple)` | float32 |
| `w_model` | `(L * d_ple, D)` | float32 |
| `norm_gain` | `(d_ple,)` | float32 |
| `ple` (result of `ple_inputs`) | `(T, L, d_ple)` | float32 |
| `ple_i` | `(T, d_ple)` | float32 |
| results of `ple_inject` and `ple_layer` | `(T, D)` | float32 |

## Example

One token ($T = 1$), $D = 2$, $L = 2$ layers, $d_{\text{ple}} = 2$, and all norm gains equal to 1:

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

For `ple_inputs`, row 1 of `table` multiplied by $\sqrt{2}$ is $(1.4142, 0 \mid 0, 2.8284)$, split into one slice per layer. The projection $h_0 W_{\text{model}}^\top = (2, 2, 0, 6)$ multiplied by $2^{-1/2}$ is $(1.4142, 1.4142 \mid 0, 4.2426)$. Layer 0's slice has mean square 2 and normalizes to $(1, 1)$. Layer 1's slice has mean square 9 and normalizes to $(0, 1.4142)$. Adding the corresponding table slices gives $(2.4142, 1)$ and $(0, 4.2426)$, and multiplying by $2^{-1/2}$ yields the result.

## Constraints

- $1 \le T \le 64$, $4 \le D \le 64$, $1 \le L \le 35$, $2 \le d_{\text{ple}} \le 16$, $V \le 64$. Token IDs lie in $[0, V)$ and may repeat.
- Entries of `h0`, `h`, and `x` have magnitudes from roughly $10^{-3}$ to 10.
- Results are compared against the reference implementation and, on a tiny configuration, against arrays computed by `Gemma4TextModel` and `Gemma4TextDecoderLayer` in Hugging Face Transformers, with relative tolerance `1e-4` and absolute tolerance `1e-5`.
- Layer $i$ of `ple_inputs` must not depend on any column of `table` or any row of `w_model` outside slice $i$.

## Why this matters

In the Gemma 4 E2B configuration ($L = 35$, $d_{\text{ple}} = 256$, $V = 262{,}144$), the packed per-layer table has width 8960 and holds 2.35 billion parameters, compared with 0.40 billion in the main embedding table of width 1536. Because it is an embedding lookup, it adds only one row read per token rather than a large matrix multiplication.

Implementation bugs here rarely crash. Splitting the packed row in the wrong order, passing layer 0's slice to every layer, or injecting before the MLP still produces plausible logits from the main residual path, silently scrambling every layer's token-specific signal.
