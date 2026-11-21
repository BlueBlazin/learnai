Implement single-head scaled dot-product self-attention with a causal mask, as a pure
function of the input and four projection matrices. Every line is a familiar formula;
the difficulty is that almost every wrong version still returns a tensor of the right
shape with plausible numbers in it.

![This problem is attention, here for one head under a causal mask.](assets/map.svg)

## Specification

```python
def causal_self_attention(x, w_q, w_k, w_v, w_o): ...
```

The input `x` holds `B` independent sequences of `T` token vectors of width `d`. For one
sequence $x_0, \dots, x_{T-1}$ (each $x_t \in \mathbb{R}^d$, a column vector):

1. **Project.** $q_t = W_q x_t$, $k_t = W_k x_t$, $v_t = W_v x_t$. The weights are
   stored in `(out, in)` layout, the layout of `nn.Linear.weight` and of model
   checkpoints: `w_q[i, j]` multiplies component `j` of the input and contributes to
   component `i` of the output. There are no biases.

2. **Score.** For query position $i$ and key position $j$, with $d_k$ the width of the
   queries and keys:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} .
$$

3. **Mask and normalize.** Position $i$ attends to positions $j \le i$ only (itself and
   the past). The attention weights are a softmax over the visible positions:

$$
a_{ij} = \begin{cases}
\dfrac{e^{s_{ij}}}{\sum_{l \le i} e^{s_{il}}} & j \le i, \\[2ex]
0 & j > i .
\end{cases}
$$

4. **Mix and project out.**

$$
y_i = W_o \sum_{j \le i} a_{ij} \, v_j .
$$

![Steps 2 to 4 for one head, read from the bottom, with the shape of the tensor on each wire. $W_o$ is applied to the output at the top.](assets/attention-steps.svg)

- `x`: float32, shape `(B, T, d)`.
- `w_q`, `w_k`: float32, shape `(d_k, d)`.
- `w_v`: float32, shape `(d_v, d)`.
- `w_o`: float32, shape `(d_out, d_v)`.
- Returns `y`: float32, shape `(B, T, d_out)`, where `y[b, i]` is $y_i$ for sequence `b`.

Two consequences of the definition are tested directly:

- **Causality.** `y[b, i]` depends only on `x[b, 0], ..., x[b, i]`. Changing a later
  token must not change it.
- **Batch independence.** `y[b]` depends only on `x[b]`.

You may use your framework's `softmax`, matrix multiplication, and `einsum`. Do not call
a built-in attention function or layer (`scaled_dot_product_attention`,
`dot_product_attention`, `MultiheadAttention`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, T, d)` | float32 |
| `w_q`, `w_k` | `(d_k, d)` | float32 |
| `w_v` | `(d_v, d)` | float32 |
| `w_o` | `(d_out, d_v)` | float32 |
| result | `(B, T, d_out)` | float32 |

## Example

One sequence of three tokens, with every projection the $2 \times 2$ identity (so
$q_t = k_t = v_t = x_t$ and $d_k = 2$):

```python
x = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]  # shape (1, 3, 2)
I = [[1.0, 0.0], [0.0, 1.0]]

causal_self_attention(x, I, I, I, I)
# [[[1.000000, 0.000000],
#   [0.330238, 0.669762],
#   [0.751745, 0.751745]]]
```

The scores $s_{ij} = x_i \cdot x_j / \sqrt{2}$ are

$$
s = \begin{bmatrix}
0.7071 & 0 & 0.7071 \\
0 & 0.7071 & 0.7071 \\
0.7071 & 0.7071 & 1.4142
\end{bmatrix} .
$$

- Token 0 sees only itself: $a_{00} = 1$ and $y_0 = x_0 = [1, 0]$.
- Token 1 sees tokens 0 and 1 with scores $[0, 0.7071]$, so $a_1 = [0.3302, 0.6698]$
  and $y_1 = 0.3302\,x_0 + 0.6698\,x_1 = [0.3302, 0.6698]$. The score $0.7071$ against
  token 2 is never used.
- Token 2 sees everything: $a_2 = [0.2483, 0.2483, 0.5035]$ and
  $y_2 = [0.7517, 0.7517]$.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN is always wrong.
- $1 \le B, T \le 16$ and $1 \le d, d_k, d_v, d_{out} \le 16$. The projections are not
  necessarily square, and $d_k$ is not necessarily equal to $d$.
- Scores satisfy $|s_{ij}| \le 500$.
- No Python loops over batch items or positions are needed.

## Why this matters

This function is the core of every decoder-only language model, and the causal mask is
what makes generation consistent with training: token $i$ was trained without seeing
tokens after $i$, and a KV cache is only valid because earlier outputs never change
when tokens are appended. Each of the classic bugs breaks that quietly. A mask applied
after the softmax still returns zeros in the right places, but the surviving weights
were normalized against future tokens, so prefix outputs change as the sequence grows
and cached decoding disagrees with a full forward pass. A missing $1/\sqrt{d_k}$
loads real weights without an error and produces fluent-looking but wrong text. A leak
across the batch makes a request's output depend on whichever other requests it was
batched with, which shows up in production as answers that cannot be reproduced.
