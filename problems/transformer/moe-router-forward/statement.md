Implement a mixture-of-experts (MoE) layer that replaces a Transformer block's single MLP with multiple expert MLPs and a learned router. For each token independently, the router scores all experts, selects the highest-scoring ones, renormalizes their probabilities, and returns the weighted sum of their outputs.

![This problem is the MLP slot of a block, filled by a router and several expert MLPs.](assets/map.svg)

## Specification

```python
def moe(x, w_router, experts_w, k): ...
```

The input `x` contains a batch of `B` sequences, each of length `T`, where every token is a vector of dimension `D`. The layer holds `E` experts, and each expert `e` is a SwiGLU MLP with hidden width `H` and its own weight matrices `w_gate[e]`, `w_up[e]`, and `w_down[e]` (each stored in `(out, in)` layout). The argument `experts_w` is the tuple `(w_gate, w_up, w_down)`, stacking all experts' weights along a leading axis of size `E`.

For a single token vector $x \in \mathbb{R}^D$, expert $e$ computes

$$
f_e(x) = W^{(e)}_{\text{down}} \Big( \operatorname{silu}\big(W^{(e)}_{\text{gate}}\, x\big) \odot W^{(e)}_{\text{up}}\, x \Big),
$$

which matches `ops.swiglu(x, w_gate[e], w_up[e], w_down[e])` from the kit.

Process each token vector $x \in \mathbb{R}^D$ independently in four steps:

1. **Route.** Project the token with the bias-free router matrix `w_router` (stored in `(out, in)` layout, where row `e` corresponds to expert `e`) to get one logit per expert. Then apply softmax across the `E` experts to obtain routing probabilities:

$$
r = W_{\text{router}}\, x \in \mathbb{R}^E, \qquad p = \operatorname{softmax}(r) .
$$

2. **Select.** Let $S$ be the set of the `k` experts with the largest logits $r_e$. Break ties in favor of the smaller expert index.

3. **Renormalize.** Keep the probabilities of the selected experts and rescale them so they sum to 1. Assign weight 0 to every unselected expert:

$$
w_e = \begin{cases} \dfrac{p_e}{\sum_{j \in S} p_j} & e \in S, \\[2ex] 0 & e \notin S . \end{cases}
$$

4. **Mix.** Return the weighted sum of the selected experts' outputs:

$$
y = \sum_{e \in S} w_e \, f_e(x) .
$$

![Every expert reads $x$, and the router's weights $w$ decide how much each expert's output counts in the sum.](assets/moe-layer.svg)

You may use dense evaluation, where every expert runs on every token and zero weights discard the unselected outputs. Only the numerical output is checked here. Dispatching each expert to only its assigned tokens is a separate problem. You may use the kit's `ops.softmax` and `ops.swiglu`, as well as your framework's sorting functions.

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

Consider one sequence of two tokens with $D = 2$, three experts with hidden width $H = 2$, and $k = 2$:

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

Each token picks its own experts. For token 0, the renormalized weights are $0.8668 / (0.8668 + 0.1173) = 0.8808$ and $0.1173 / (0.8668 + 0.1173) = 0.1192$. For token 1, expert 2 has the highest logit, while experts 0 and 1 tie for second with a logit of $-1$. Tie-breaking selects the lower index (expert 0), giving it weight $0.0453 / (0.9094 + 0.0453) = 0.0474$.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- $1 \le B \le 4$, $1 \le T \le 8$, $2 \le D \le 16$, $1 \le E \le 8$, $1 \le k \le E$, $2 \le H \le 16$.
- Router logits satisfy $|r_e| \le 200$.
- In every test, the $k$-th and $(k+1)$-th largest logits of a token are either exactly equal (inputs made of small integers, where float32 arithmetic is exact) or differ by more than $10^{-3}$, so floating-point rounding cannot change which experts are selected.

## Why this matters

MoE layers let modern language models scale up total parameter count without increasing per-token compute. Mixtral 8x7B holds 8 expert MLPs per layer and activates 2 per token, while Qwen3-30B-A3B holds 128 and activates 8. Because a small router decides which MLP weights each token uses, routing bugs cause large accuracy drops rather than minor numerical drift.

Unnormalized weights shrink the residual update by a different amount at every layer. Routing per sequence or per batch instead of per token sends tokens through experts that were never trained to handle them. For an inference engine, this dense reference also serves as the exact specification that a fast sparse kernel must match.
