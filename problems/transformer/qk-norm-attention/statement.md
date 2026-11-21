Implement causal attention with QK-norm: before the dot product, every query and every
key is RMS-normalized over its head dimension and multiplied by a learned gain, and
only then rotated by RoPE. Three choices decide the result: which axis the
normalization runs over, what it is applied to, and where it sits relative to the
rotation.

![This problem is attention with RoPE, where queries and keys are normalized before they are rotated.](assets/map.svg)

## Specification

```python
def qk_norm_attention(q, k, v, q_weight, k_weight, cos, sin, eps=1e-6): ...
```

`q`, `k`, and `v` are already projected and split into heads: shape `(B, H, T, d)`,
with `d` the width of one head (even). For one batch item and one head, write
$q_t, k_t, v_t \in \mathbb{R}^d$ for the vectors of token $t$.

1. **Normalize the queries and the keys.** Each vector is divided by its own root mean
   square over its $d$ components, then multiplied elementwise by a gain. The epsilon
   is added to the mean square, inside the square root:

$$
\hat{q}_t = \frac{q_t}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1} q_{t,i}^2 + \varepsilon}} \odot g^{q},
\qquad
\hat{k}_t = \frac{k_t}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1} k_{t,i}^2 + \varepsilon}} \odot g^{k}.
$$

   The gains $g^q$ = `q_weight` and $g^k$ = `k_weight` have shape `(d,)` and are shared
   by all heads, positions, and batch items. Every vector `q[b, h, t]` has its own
   root mean square. The values are **not** normalized.

2. **Rotate the normalized queries and keys (RoPE).** `cos` and `sin` have shape
   `(T, d/2)` and hold the cosine and sine of the rotation angle of each pair of
   adjacent dimensions at each token's position. For pair $i = 0, \dots, d/2 - 1$:

$$
\begin{aligned}
\tilde{q}_{t,2i} &= \hat{q}_{t,2i}\,\cos[t, i] - \hat{q}_{t,2i+1}\,\sin[t, i], \\
\tilde{q}_{t,2i+1} &= \hat{q}_{t,2i}\,\sin[t, i] + \hat{q}_{t,2i+1}\,\cos[t, i],
\end{aligned}
$$

   and the same for $\tilde{k}_t$ from $\hat{k}_t$, with the same tables. The values
   are not rotated.

3. **Attend causally**, with the usual scale:

$$
s_{ij} = \frac{\tilde{q}_i \cdot \tilde{k}_j}{\sqrt{d}}, \qquad
a_{ij} = \begin{cases}
\dfrac{e^{s_{ij}}}{\sum_{l \le i} e^{s_{il}}} & j \le i, \\[2ex]
0 & j > i,
\end{cases}
\qquad
y_i = \sum_{j \le i} a_{ij}\, v_j .
$$

- `q`, `k`, `v`: float32, shape `(B, H, T, d)`, `d` even.
- `q_weight`, `k_weight`: float32, shape `(d,)`.
- `cos`, `sin`: float32, shape `(T, d/2)`. The same tables apply to every batch item
  and head. They are given; you do not compute angles.
- `eps`: a positive Python float, $\varepsilon$ above.
- Returns `y`: float32, shape `(B, H, T, d)`.

You may use your framework's `softmax`, matrix multiplication, `einsum`, and RMS
normalization. Do not call a built-in attention function or layer
(`scaled_dot_product_attention`, `dot_product_attention`, `MultiheadAttention`, and the
like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, `k`, `v` | `(B, H, T, d)`, `d` even | float32 |
| `q_weight`, `k_weight` | `(d,)` | float32 |
| `cos`, `sin` | `(T, d/2)` | float32 |
| result | `(B, H, T, d)` | float32 |

## Example

One batch item, one head, two tokens, $d = 2$ (a single rotation pair), at positions 0
and 1 with one radian per position:

```python
q = [[[[3.0, 4.0], [4.0, -3.0]]]]  # shape (1, 1, 2, 2)
k = [[[[1.0, 0.0], [20.0, 20.0]]]]
v = [[[[1.0, 0.0], [0.0, 1.0]]]]
q_weight = [1.0, 0.5]
k_weight = [2.0, 1.0]
cos = [[1.0], [0.540302]]  # cos(0), cos(1); shape (2, 1)
sin = [[0.0], [0.841471]]  # sin(0), sin(1)

qk_norm_attention(q, k, v, q_weight, k_weight, cos, sin)
# [[[[1.000000, 0.000000],
#    [0.653979, 0.346021]]]]
```

- **Normalize.** Both queries have root mean square $\sqrt{(9 + 16)/2} = 3.5355$, so
  $\hat{q}_0 = [0.8485, 1.1314] \odot [1, 0.5] = [0.8485, 0.5657]$ and
  $\hat{q}_1 = [1.1314, -0.4243]$. The keys have root mean squares $0.7071$ and $20$:
  $\hat{k}_0 = [1.4142, 0] \odot [2, 1] = [2.8284, 0]$ and
  $\hat{k}_1 = [1, 1] \odot [2, 1] = [2, 1]$. The second key was twenty times longer
  than the first; after the normalization they are the same size.
- **Rotate.** Token 0 is at angle 0 and unchanged. Token 1 turns by one radian:
  $\tilde{q}_1 = [0.9683, 0.7228]$ and $\tilde{k}_1 = [0.2391, 2.2232]$.
- **Attend.** Token 0 sees only itself, so $y_0 = v_0 = [1, 0]$. Token 1 has the
  scores $s_{10} = \tilde{q}_1 \cdot \tilde{k}_0 / \sqrt{2} = 1.9366$ and
  $s_{11} = \tilde{q}_1 \cdot \tilde{k}_1 / \sqrt{2} = 1.3000$, hence the weights
  $[0.6540, 0.3460]$ and $y_1 = 0.6540\,v_0 + 0.3460\,v_1$.

Without the normalization the same inputs give $s_{10} = 3.31$ and $s_{11} = 14.14$,
and token 1 puts weight $0.99998$ on the long key.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le B, H, T \le 8$; $d$ is even, $2 \le d \le 16$.
- The entries of `q`, `k`, and `v` are at most $10^3$ in absolute value, so squares do
  not overflow, and every query and key vector has a root mean square of at least
  $10^{-3}$. The gains lie in $[0.1, 3]$ and $10^{-6} \le$ `eps` $\le 1$.
- Each row `(cos[t, i], sin[t, i])` is the cosine and sine of one angle.
- No Python loops over batch items, heads, positions, or pairs are needed.

## Why this matters

In plain attention the size of a score grows with the lengths of the query and the
key, so a few long vectors can saturate the softmax. QK-norm removes the lengths from
the dot product and lets a learned gain set the sharpness instead. Qwen3 applies it in every layer (`q_norm` and
`k_norm` in its checkpoints), and Gemma 3 adopted it in place of the logit soft-capping
of Gemma 2. For an inference engine it is two extra small tensors per layer whose
names say nothing about how to apply them. Normalize over the whole hidden vector
instead of per head, rotate before normalizing, or normalize the values as well, and
the checkpoint still loads and still generates fluent text, only worse, with nothing
pointing to the attention layer.
