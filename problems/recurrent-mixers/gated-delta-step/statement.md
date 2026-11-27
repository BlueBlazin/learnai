Implement one token step of the gated delta rule, the recurrent token mixer used in the linear-attention layers of Qwen3.5. Each head maintains a fixed-size matrix memory that maps keys to values, decays slightly on every token, and overwrites what it holds under a key instead of only adding to it.

## Specification

```python
def gdn_gates(a, b, A_log, dt_bias): ...  # -> (g, beta)
def gdn_step(S, q, k, v, g, beta): ...  # -> (y, S)
```

A layer processes a sequence of $T$ tokens across $H$ independent heads. Each head $h$ holds a memory matrix $S_h \in \mathbb{R}^{D_v \times D_k}$ that maps a key vector $k \in \mathbb{R}^{D_k}$ to a predicted value vector $S_h k \in \mathbb{R}^{D_v}$.

**`gdn_gates`** converts two raw scalar inputs per token and head (`a` and `b`) into the decay gate $g$ and the write strength $\beta$. Using the per-head layer parameters `A_log` and `dt_bias`, the gates for token $t$ and head $h$ are:

$$
g_{t,h} = \exp\!\bigl(-e^{A_{\log,h}} \cdot \operatorname{softplus}(a_{t,h} + \texttt{dt\_bias}_h)\bigr),
\qquad
\beta_{t,h} = \sigma(b_{t,h}) = \frac{1}{1 + e^{-b_{t,h}}},
$$

where $\operatorname{softplus}(x) = \log(1 + e^{x})$. Because $e^{A_{\log,h}} > 0$ and $\operatorname{softplus}(\cdot) > 0$, the exponent is always negative and $g_{t,h}$ lies in $(0, 1)$ in exact arithmetic.

**`gdn_step`** advances all $H$ heads by one token. For each head (dropping the head index), given query $q \in \mathbb{R}^{D_k}$, key $k \in \mathbb{R}^{D_k}$, value $v \in \mathbb{R}^{D_v}$, and scalar gates $g, \beta \in [0, 1]$, it applies three operations in strict order:

$$
\begin{aligned}
\text{decay:} \quad & \bar S = g\,S \\
\text{write:} \quad & S' = \bar S + \beta\,(v - \bar S\,k)\,k^\top \\
\text{read:}  \quad & y = S'\,q
\end{aligned}
$$

First, the existing memory $S$ is scaled by $g$. Next, the write computes the prediction error $v - \bar S k$ against the decayed memory and adds a rank-one update along $k$ scaled by $\beta$. Finally, the output $y$ is read from the updated memory $S'$ so the token can see its own write.

- `q` and `k` are used as given. Do not normalize or rescale them inside `gdn_step` (the surrounding layer handles that before calling the step).
- `gdn_step` takes `S` as its first argument and returns `(y, S')` with the updated memory as the second result.
- Do not modify any input tensor, including `S`, in place.
- Do not call the kit's `gdn_gates_ref`, `gdn_step_ref`, or `ops.gated_delta_layer`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `a`, `b` | `(T, H)` | float32 |
| `A_log`, `dt_bias` | `(H,)` | float32 |
| `g`, `beta` returned by `gdn_gates` | `(T, H)` | float32 |
| `S` (argument and result) | `(H, Dv, Dk)` | float32 |
| `q`, `k` | `(H, Dk)` | float32 |
| `v`, `y` | `(H, Dv)` | float32 |
| `g`, `beta` passed to `gdn_step` | `(H,)` | float32 |

## Example

Computing gates for one token and two heads:

```python
gdn_gates(a=[[0.0, 2.0]], b=[[0.0, -1.0]], A_log=[0.0, -1.0], dt_bias=[0.0, 1.0])
# g    = [[0.5, 0.325817]]
# beta = [[0.5, 0.268941]]
```

For head 0, $\operatorname{softplus}(0) = \ln 2$ and $e^{0} = 1$, which gives $g = e^{-\ln 2} = 0.5$. For head 1, $\operatorname{softplus}(3) = 3.0486$ and $e^{-1} = 0.3679$, which gives $g = e^{-1.1215} = 0.3258$.

Running one step for one head with $D_v = D_k = 2$, a unit-length key, $g = 0.5$, and $\beta = 1$:

```python
S = [[[1.0, 0.0], [0.0, 2.0]]]
q, k, v = [[1.0, 0.0]], [[0.6, 0.8]], [[1.0, -1.0]]

gdn_step(S, q, k, v, g=[0.5], beta=[1.0])
# y = [[0.92, -1.08]]
# S = [[[0.92, 0.56], [-1.08, -0.44]]]
```

![The example step on $S$: decay by $g$, write toward $v$ under $k$, then read with $q$.](assets/delta-step.svg)

Decaying $S$ by $0.5$ gives $\bar S = [[0.5, 0], [0, 1]]$, which predicts $\bar S k = (0.3, 0.8)$ for the key. The correction is $v - \bar S k = (0.7, -1.8)$, and adding its outer product with $k$ produces $S'$. Querying $S'$ with $k$ yields $S' k = (0.552 + 0.448,\; -0.648 - 0.352) = (1, -1) = v$, so the memory returns the exact new value under this key regardless of its previous contents. Because $q = (1, 0)$, the output $y$ is the first column of $S'$.

With $\beta = 0.5$, the same call moves halfway toward $v$, returning `y = [[0.71, -0.54]]` and `S = [[[0.71, 0.28], [-0.54, 0.28]]]`. For this state, $S' k = (0.65, -0.1)$, which is the midpoint of $(0.3, 0.8)$ and $v$.

## Constraints

- $1 \le T \le 8$, $1 \le H \le 4$, $2 \le D_k \le 16$, and $2 \le D_v \le 8$.
- In `gdn_gates`, $|a + \texttt{dt\_bias}| \le 21$ and $|b| \le 20$ (so $\log(1 + e^{x})$ does not overflow), and $0.01 \le e^{A_{\log}} \le 16$. In exact arithmetic $0 < g < 1$, though float32 rounding may produce `0` or `1`.
- In `gdn_step`, $0 \le g \le 1$ and $0 \le \beta \le 1$. The boundary values $g = 0$, $g = 1$, $\beta = 0$, and $\beta = 1$ all occur. Keys and queries may have any length, and the initial memory may be all zeros.
- Outputs are checked against the reference to a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. A six-token sequence is also checked against the delta rule in Hugging Face Transformers to the same tolerance.
- Two algebraic properties are tested to the same tolerance. With $\beta = 1$ and a unit-length key, $S' k$ must equal $v$ for every $g$. With $g = 1$, a write must leave $S k'$ unchanged for every $k'$ orthogonal to $k$.
- No Python loop over heads is needed.

## Why this matters

Eighteen of the 24 layers in Qwen3.5-0.8B mix tokens with this recurrence instead of softmax attention. The matrix $S$ is the entire state such a layer keeps for the prefix, so memory usage stays constant at any context length.

Getting the order of the three operations wrong produces a silently different model. Reading before writing hides a token from itself, decaying after writing shrinks every value before its first use, and omitting the $-\bar S k$ correction turns the rule into plain additive linear attention that saturates on repeated keys. None of these bugs raises an error, and all of them compound across tokens.
