Implement the expert router of DeepSeek-V3 and its successors, and the layer around it:
experts are scored independently, chosen by a biased score, weighted by the unbiased
one, and joined by a shared expert that every token uses.

![This problem is the feed-forward layer, built here from routed experts and one shared expert.](assets/map.svg)

## Specification

```python
def route(x, w_router, bias, k, scale, score_fn="sigmoid"): ...  # -> (indices, weights)
def moe_shared(x, w_router, bias, experts_w, shared_w, k, scale, score_fn="sigmoid"): ...
```

`x` has shape `(N, D)`, one row per token. There are $E$ routed experts. For **each
token** $x \in \mathbb{R}^D$ independently:

1. **Score.** The router is a linear layer without bias; each logit becomes a score on
   its own:

$$
r = W_{\text{router}}\, x \in \mathbb{R}^E, \qquad
s_e = \begin{cases}
\sigma(r_e) = \dfrac{1}{1 + e^{-r_e}} & \texttt{score\_fn} = \texttt{"sigmoid"}, \\[2ex]
\sqrt{\operatorname{softplus}(r_e)} = \sqrt{\ln(1 + e^{r_e})} & \texttt{score\_fn} = \texttt{"sqrtsoftplus"} .
\end{cases}
$$

2. **Select.** The chosen experts $e_1, \dots, e_k$ are the $k$ experts with the
   largest **biased** scores $s_e + b_e$, where $b$ is `bias`, listed from the largest
   down. Equal biased scores are ranked by expert index, lowest first.

3. **Weight.** The weights use the **unbiased** scores, normalized among the chosen
   experts and then scaled:

$$
w_j = \frac{s_{e_j}}{\displaystyle\sum_{i=1}^{k} s_{e_i} + 10^{-20}} \cdot \texttt{scale} .
$$

`route` returns `(indices, weights)`: $e_1, \dots, e_k$ and $w_1, \dots, w_k$ for every
token, both of shape `(N, k)`.

**`moe_shared`** is the layer. Routed expert $e$ is the SwiGLU MLP $f_e$ with the
weights `w_gate[e]`, `w_up[e]`, `w_down[e]` (`ops.swiglu` from the kit, as in
`moe-router-forward`), and $f_{\text{shared}}$ is one more SwiGLU MLP with the weights
`shared_w`:

$$
y = \sum_{j=1}^{k} w_j\, f_{e_j}(x) \;+\; f_{\text{shared}}(x) .
$$

The shared expert has weight 1: it is not selected, not normalized, and not multiplied
by `scale`.

Both functions raise `ValueError` for a `score_fn` other than the two names.

- `x`: float32, shape `(N, D)`. `w_router`: float32, shape `(E, D)`, in `(out, in)`
  layout. `bias`: float32, shape `(E,)`; may be negative.
- `k`: a Python int, $1 \le k \le E$. `scale`: a Python float greater than 0.
- `experts_w`: a tuple `(w_gate, w_up, w_down)` of shapes `(E, H, D)`, `(E, H, D)`,
  `(E, D, H)`. `shared_w`: a tuple of shapes `(H_s, D)`, `(H_s, D)`, `(D, H_s)`.
- `route` returns `indices` (integer) and `weights` (float32), both `(N, k)`.
  `moe_shared` returns float32, shape `(N, D)`.

Dense evaluation (every expert on every token) is fine here. You may use your
framework's `sigmoid`, `softplus`, and sorting functions, and the kit's `ops.swiglu`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(N, D)` | float32 |
| `w_router` | `(E, D)` | float32 |
| `bias` | `(E,)` | float32 |
| `indices` | `(N, k)` | integer |
| `weights` | `(N, k)` | float32 |
| `experts_w` | `(E, H, D)`, `(E, H, D)`, `(E, D, H)` | float32 |
| `shared_w` | `(H_s, D)`, `(H_s, D)`, `(D, H_s)` | float32 |
| result of `moe_shared` | `(N, D)` | float32 |

## Example

Two tokens with $D = 2$, four experts, $k = 2$, `scale = 2.5`, and a bias on expert 2:

```python
x = [[2.0, 0.0], [1.0, 1.0]]
w_router = [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [-1.0, 0.0]]
bias = [0.0, 0.0, 0.3, 0.0]

route(x, w_router, bias, 2, 2.5)
# indices [[0, 2], [2, 0]]
# weights [[1.594726, 0.905274], [1.25, 1.25]]
```

| | token 0: $x = [2, 0]$ | token 1: $x = [1, 1]$ |
|---|---|---|
| logits $r$ | $[2, 1, 0, -2]$ | $[1, 0.5, 1, -1]$ |
| scores $s = \sigma(r)$ | $[0.8808, 0.7311, 0.5, 0.1192]$ | $[0.7311, 0.6225, 0.7311, 0.2689]$ |
| biased $s + b$ | $[0.8808, 0.7311, 0.8, 0.1192]$ | $[0.7311, 0.6225, 1.0311, 0.2689]$ |
| chosen, best first | 0, 2 | 2, 0 |
| weights | $1.5947,\ 0.9053$ | $1.25,\ 1.25$ |

For token 0 the bias lifts expert 2 over expert 1, and its weight still comes from its
own score: $2.5 \cdot 0.5 / (0.8808 + 0.5) = 0.9053$. With `score_fn="sqrtsoftplus"`
the scores of token 0 are $[1.4584, 1.1460, 0.8326, 0.3563]$, the bias no longer lifts
expert 2, and the result is indices `[[0, 1], [2, 0]]` and weights
`[[1.399951, 1.100049], [1.25, 1.25]]`.

The layer, with `I` the identity and a shared expert of hidden width 1:

```python
I = [[1.0, 0.0], [0.0, 1.0]]
w_gate = [I, [[2.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, -1.0]]]
w_up = [I, I, I, I]
w_down = [I, I, [[1.0, 1.0], [0.0, 1.0]], I]
shared_w = ([[1.0, 1.0]], [[1.0, -1.0]], [[1.0], [2.0]])

moe_shared(x, w_router, bias, (w_gate, w_up, w_down), shared_w, 2, 2.5)
# [[9.141706, 7.046376],
#  [2.741470, 1.827646]]
```

Token 0: $f_0(x) = [3.5232, 0]$, $f_2(x) = [0, 0]$, $f_{\text{shared}}(x) =
[3.5232, 7.0464]$, so $y = 1.5947 \cdot [3.5232, 0] + [3.5232, 7.0464]$. Token 1:
$f_2(x) = [1.4621, 0.7311]$, $f_0(x) = [0.7311, 0.7311]$, and the shared expert returns
0, so $y = 1.25 \cdot [2.1932, 1.4621]$.

## Constraints

- $1 \le N \le 8$, $2 \le D \le 16$, $1 \le E \le 384$, $1 \le k \le \min(E, 8)$,
  $0.5 \le \texttt{scale} \le 2.5$.
- `indices` are compared exactly; `weights` and the output of `moe_shared` to a
  relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. For the routing
  parameters of GLM-5.3 and DeepSeek-V4 the weights are also compared with those
  returned by the router modules of Hugging Face Transformers.
- Router logits satisfy $|r_e| \le 12$ with sigmoid scores and $|r_e| \le 5$ with
  `sqrtsoftplus`, because float32 `softplus` implementations disagree for very
  negative logits. Two kinds of test go beyond that: sigmoid scores with logits
  between $-32$ and $-30$, and logits of $-120$, for which both score functions
  return exactly 0 in float32 and the weights must be 0.
- In every test, two biased scores of a token are either exactly equal (inputs built
  from small integers, halves, and constant rows) or more than $10^{-3}$ apart among
  the $k + 1$ largest. The exception is the test with logits between $-32$ and
  $-30$, where the bias is zero and the scores, near $10^{-13}$, differ from one
  another by more than 20 percent.

## Why this matters

GLM-5.3 routes each token to 8 of 256 experts this way with a scale of 2.5, and
DeepSeek-V4 to 6 of 384 with `sqrtsoftplus` scores; both add one shared expert. A
router ported from a softmax model tends to apply the scale before the normalization,
where it cancels: every routed expert then contributes 2.5 times too little relative
to the shared expert, and the model still produces fluent text. Dropping the bias, or
letting it into the weights, changes which experts run or how much they count.
Neither raises an error.
