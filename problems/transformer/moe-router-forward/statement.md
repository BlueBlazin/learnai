Implement a mixture-of-experts (MoE) layer: in place of one MLP, the block holds
several, and a small router sends each token to a few of them. The arithmetic is a
softmax, a top-k, and a weighted sum; the care is in doing them per token, in the
specified order, with weights that sum to one.

## Specification

```python
def moe(x, w_router, experts_w, k): ...
```

The layer has $E$ experts. Expert $e$ is a SwiGLU MLP with its own weights,

$$
f_e(x) = W^{(e)}_{\text{down}} \Big( \operatorname{silu}\big(W^{(e)}_{\text{gate}}\, x\big) \odot W^{(e)}_{\text{up}}\, x \Big),
$$

which is `ops.swiglu(x, w_gate[e], w_up[e], w_down[e])` from the kit. For **each token
vector** $x \in \mathbb{R}^D$ independently (each position of each sequence):

1. **Route.** The router is a linear layer without bias. It gives one logit per expert,
   and a softmax over the experts turns them into probabilities:

$$
r = W_{\text{router}}\, x \in \mathbb{R}^E, \qquad p = \operatorname{softmax}(r) .
$$

2. **Select.** Let $S$ be the set of the $k$ experts with the largest logits $r_e$.
   Experts with equal logits are ranked by index, lowest first: of two experts with
   the same logit, the one with the smaller index is selected first.

3. **Renormalize.** The selected experts keep their probabilities, rescaled to sum
   to 1; every other expert gets weight 0:

$$
w_e = \begin{cases} \dfrac{p_e}{\sum_{j \in S} p_j} & e \in S, \\[2ex] 0 & e \notin S . \end{cases}
$$

4. **Mix.** The output is the weighted sum of the selected experts' outputs:

$$
y = \sum_{e \in S} w_e \, f_e(x) .
$$

Arguments and result:

- `x`: float32, shape `(B, T, D)`.
- `w_router`: float32, shape `(E, D)`, in `(out, in)` layout: row `e` produces the
  logit of expert `e`.
- `experts_w`: a tuple `(w_gate, w_up, w_down)` of three float32 tensors that stack
  the weights of all experts along a leading axis: `w_gate` and `w_up` have shape
  `(E, H, D)` and `w_down` has shape `(E, D, H)`. Each `w_gate[e]`, `w_up[e]`,
  `w_down[e]` is in `(out, in)` layout.
- `k`: a Python int with $1 \le k \le E$.
- Returns `y`: float32, shape `(B, T, D)`.

You may evaluate every expert on every token and let the zero weights discard the
unselected ones (dense evaluation). Only the result is checked here; running each
expert on just its own tokens is a later problem. You may use the kit's `ops.softmax`
and `ops.swiglu`, and your framework's sorting functions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, T, D)` | float32 |
| `w_router` | `(E, D)` | float32 |
| `experts_w[0]` (`w_gate`) | `(E, H, D)` | float32 |
| `experts_w[1]` (`w_up`) | `(E, H, D)` | float32 |
| `experts_w[2]` (`w_down`) | `(E, D, H)` | float32 |
| `k` | Python int, $1 \le k \le E$ | |
| result | `(B, T, D)` | float32 |

## Example

One sequence of two tokens with $D = 2$, three experts of hidden width $H = 2$, and
$k = 2$:

```python
x = [[[2.0, 0.0], [-1.0, -1.0]]]  # shape (1, 2, 2)
w_router = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]  # shape (3, 2)
I = [[1.0, 0.0], [0.0, 1.0]]
w_gate = [I, [[2.0, 0.0], [0.0, 2.0]], [[-1.0, 0.0], [0.0, -1.0]]]  # shape (3, 2, 2)
w_up = [I, I, I]
w_down = [I, [[0.0, 1.0], [1.0, 0.0]], I]

moe(x, w_router, (w_gate, w_up, w_down), 2)
# [[[ 3.103214,  0.936471],
#   [-0.683633, -0.683633]]]
```

| | token 0: $x = [2, 0]$ | token 1: $x = [-1, -1]$ |
|---|---|---|
| router logits $r$ | $[2, 0, -2]$ | $[-1, -1, 2]$ |
| $p = \operatorname{softmax}(r)$ | $[0.8668, 0.1173, 0.0159]$ | $[0.0453, 0.0453, 0.9094]$ |
| selected $S$ | experts 0 and 1 | experts 2 and 0 |
| weights $w$ | $[0.8808, 0.1192, 0]$ | $[0.0474, 0, 0.9526]$ |
| $f_0(x)$ | $[3.5232, 0]$ | $[0.2689, 0.2689]$ |
| $f_1(x)$ | $[0, 7.8561]$ | $[0.2384, 0.2384]$ |
| $f_2(x)$ | $[-0.4768, 0]$ | $[-0.7311, -0.7311]$ |
| $y$ | $[3.1032, 0.9365]$ | $[-0.6836, -0.6836]$ |

The two tokens of the same sequence use different experts. For token 0 the weights are
$0.8668 / (0.8668 + 0.1173) = 0.8808$ and $0.1173 / (0.8668 + 0.1173) = 0.1192$. For
token 1, expert 2 is first, and experts 0 and 1 tie for second place with a logit of
$-1$ each: the lower index, expert 0, is selected, with weight
$0.0453 / (0.9094 + 0.0453) = 0.0474$.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`.
- $1 \le B \le 4$, $1 \le T \le 8$, $2 \le D \le 16$, $1 \le E \le 8$,
  $1 \le k \le E$, $2 \le H \le 16$.
- Router logits satisfy $|r_e| \le 200$.
- In every test, the $k$-th and $(k+1)$-th largest logits of a token are either
  exactly equal (inputs made of small integers, for which float32 arithmetic is exact)
  or differ by more than $10^{-3}$, so rounding cannot change which experts are
  selected.

## Why this matters

MoE is how recent open models grow their parameter count without growing the cost per
token: Mixtral 8x7B holds 8 expert MLPs per layer and runs 2 of them for each token,
and Qwen3-30B-A3B holds 128 and runs 8. The router is a few thousand parameters that
decide which few of the model's MLP weights a token touches, so a routing bug is not a
small numerical error. Weights that do not sum to 1 rescale the residual update of
every layer, and a selection made per sequence or per batch instead of per token sends
most tokens through experts that were never trained on them. For an inference engine,
this function is also the specification that the fast sparse version, which gathers
each expert's tokens and runs only those, has to reproduce exactly.
