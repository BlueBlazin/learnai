RoPE exists in two layouts that are the same mathematics and different code: one
rotates adjacent pairs of features, the other pairs feature $i$ with feature
$i + D/2$. A checkpoint works with exactly one of them, and nothing in the file says
which. Implement both, and the permutation of the query and key projection weights
that turns a model trained for the first into a model for the second with identical
attention scores.

![This problem is RoPE, in its two layouts of the feature pairs.](assets/map.svg)

## Specification

```python
def rope_interleaved(x, pos, theta=10000.0): ...
def rope_half(x, pos, theta=10000.0): ...
def permute_qk(w, d_head, n_heads): ...
```

`x` has shape `(B, H, T, D)`: batch, heads, sequence, head dimension, with `D` even.
`pos` has shape `(T,)`; `pos[t]` is the absolute position $p$ of the token at sequence
index `t`. Both layouts use the same $D/2$ frequencies and the same counterclockwise
rotation of a pair $(a, b)$ by the angle $p\,\theta_i$:

$$
\theta_i = \texttt{theta}^{-2i/D}, \qquad
\begin{bmatrix} a' \\ b' \end{bmatrix}
=
\begin{bmatrix}
\cos p\theta_i & -\sin p\theta_i \\
\sin p\theta_i & \phantom{-}\cos p\theta_i
\end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix},
\qquad i = 0, \dots, D/2 - 1 .
$$

They differ only in which two features form pair $i$:

| layout | pair $i$ is $(a, b) =$ | used by |
|---|---|---|
| **interleaved**, `rope_interleaved` | $(x_{2i},\; x_{2i+1})$ | the RoFormer paper; Meta's Llama reference code, which views each adjacent pair as one complex number |
| **half-split**, `rope_half` | $(x_{i},\; x_{i + D/2})$ | Hugging Face Transformers (`rotate_half`), including its Llama and Granite models |

The rotated members go back to the features they came from. `rope_interleaved` is
the function `apply_rope` of the `rope` problem.

![A head of width 8, where pair $i$ has the members $a_i$ and $b_i$. The lines show where `permute_qk` moves each feature.](assets/layouts.svg)

- `x`: float32, shape `(B, H, T, D)`, `D` even.
- `pos`: integer tensor, shape `(T,)`, values $\ge 0$; not necessarily starting at 0,
  contiguous, sorted, or distinct.
- `theta`: a Python float greater than 1 (the RoPE base).
- Both functions return float32, shape `(B, H, T, D)`.

**`permute_qk(w, d_head, n_heads)`** converts one projection weight. `w` is a query or
key projection in `(out, in)` layout, shape `(n_heads * d_head, d_in)`, of a model
that applies `rope_interleaved`. Its output features are split into heads in the
usual way: head $h$ owns rows $h \cdot d_{\text{head}}$ to
$(h + 1) \cdot d_{\text{head}} - 1$. The function returns the weight `w2` of the same
shape with

$$
\texttt{w2}\big[\,h\,d_{\text{head}} + c\,\tfrac{d_{\text{head}}}{2} + i\,\big]
= \texttt{w}\big[\,h\,d_{\text{head}} + 2i + c\,\big],
\qquad c \in \{0, 1\},\quad 0 \le i < \tfrac{d_{\text{head}}}{2},\quad 0 \le h < n_{\text{heads}} .
$$

Within each head, the first members of all pairs (the even rows) come first, in
order, and the second members (the odd rows) follow. Rows never leave their head,
and the input axis is untouched.

This is the conversion from interleaved to half-split: for every input, the head
vectors `q2`, `k2` computed with the permuted query and key weights satisfy

$$
\big\langle \texttt{rope\_half}(q_2, m),\ \texttt{rope\_half}(k_2, n) \big\rangle
= \big\langle \texttt{rope\_interleaved}(q, m),\ \texttt{rope\_interleaved}(k, n) \big\rangle ,
$$

so the attention scores, and with them the model's output, are unchanged. `n_heads`
is the number of heads in the matrix at hand: with grouped-query attention the key
projection has fewer heads than the query projection.

Do not call a built-in RoPE (`mlx.nn.RoPE`, `mx.fast.rope`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, H, T, D)`, `D` even | float32 |
| `pos` | `(T,)` | integer |
| result of `rope_interleaved`, `rope_half` | `(B, H, T, D)` | float32 |
| `w` | `(n_heads * d_head, d_in)`, `d_head` even | float32 |
| result of `permute_qk` | `(n_heads * d_head, d_in)` | float32 |

## Example

One vector with `D = 4` at position 1, and `theta = 100`, so
$\theta = [1, 0.1]$:

```python
x = [[[[1.0, 0.0, 1.0, 0.0]]]]  # shape (1, 1, 1, 4)

rope_interleaved(x, [1], 100.0)  # [[[[ 0.540302, 0.841471, 0.995004, 0.099833]]]]
rope_half(x, [1], 100.0)  # [[[[-0.301169, 0.000000, 1.381773, 0.000000]]]]
```

Interleaved: the pairs are $(x_0, x_1) = (1, 0)$, turned by 1 radian to
$(\cos 1, \sin 1)$, and $(x_2, x_3) = (1, 0)$, turned by 0.1 radian to
$(\cos 0.1, \sin 0.1)$. Half-split: the pairs are $(x_0, x_2) = (1, 1)$, turned by 1
radian to $(\cos 1 - \sin 1,\; \sin 1 + \cos 1) = (-0.3012, 1.3818)$, which go back to
features 0 and 2, and $(x_1, x_3) = (0, 0)$. The same input, two different results.

Move the features of `x` into half-split order first, $[x_0, x_2, x_1, x_3] =
[1, 1, 0, 0]$, and the two layouts agree up to the same reordering:

```python
rope_half([[[[1.0, 1.0, 0.0, 0.0]]]], [1], 100.0)  # [[[[0.540302, 0.995004, 0.841471, 0.099833]]]]
```

`permute_qk` does that reordering to the rows of a weight. With two heads of width 4
and a weight whose row $r$ holds the number $r$:

```python
w = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]  # shape (8, 1)
permute_qk(w, d_head=4, n_heads=2)
# [[0.0], [2.0], [1.0], [3.0], [4.0], [6.0], [5.0], [7.0]]
```

With one head of width 8 the row order becomes `0, 2, 4, 6, 1, 3, 5, 7`.

## Constraints

- $1 \le B, H, T \le 8$; $D$ and `d_head` are even, between 2 and 32;
  $10 \le \texttt{theta} \le 10^7$; $0 \le$ `pos[t]` $< 2048$; `n_heads` $\le 8$ and
  `d_in` $\le 64$.
- The rotations are compared with the reference. When every position is below 64:
  relative tolerance `1e-4`, absolute tolerance `1e-4`. With larger positions:
  relative tolerance `1e-3`, absolute tolerance `5e-3` (the angle is a float32
  product, and its error grows with the position).
- `permute_qk` only moves rows and is compared exactly.
- Attention scores before and after conversion are compared to a tolerance of `1e-3`.
- No Python loops over positions, pairs, or heads are needed.

## Why this matters

Meta released Llama with interleaved RoPE; Hugging Face Transformers implements
half-split RoPE. The Llama checkpoints on the Hub therefore do not contain Meta's
`wq` and `wk` matrices: the conversion script permuted their rows, with exactly this
function, so that the half-split code reproduces the original model. Every engine
that loads those files inherits the question. One whose kernel rotates adjacent pairs
has to undo the permutation when it converts the checkpoint, and one that guesses
wrong in either direction gets a model that loads, runs, and attends to the wrong
places: the scores are no longer a function of relative position at the right
frequencies, and the text degrades without any error. Models trained directly in
Transformers, such as Granite, are half-split from the start and must not be
permuted at all. The layout is part of the model, and the only place it is written
down is the code that produced the weights.
