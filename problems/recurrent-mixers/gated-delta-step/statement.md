Implement one token of the gated delta rule, the recurrence inside the linear-attention
layers of Qwen3.5: a matrix per head that maps keys to values, fades a little with
every token, and overwrites what it holds under a key instead of adding to it.

## Specification

```python
def gdn_gates(a, b, A_log, dt_bias): ...  # -> (g, beta)
def gdn_step(S, q, k, v, g, beta): ...  # -> (y, S)
```

There are $H$ heads. Head $h$ owns a memory $S_h \in \mathbb{R}^{D_v \times D_k}$, so
that $S_h\,k$ is the value the memory returns for a key $k \in \mathbb{R}^{D_k}$.

**`gdn_gates`** turns two raw numbers per token and head into the two gates of the
rule. For token $t$ and head $h$:

$$
g_{t,h} = \exp\!\bigl(-e^{A_{\log,h}} \cdot \operatorname{softplus}(a_{t,h} + \texttt{dt\_bias}_h)\bigr),
\qquad
\beta_{t,h} = \sigma(b_{t,h}) = \frac{1}{1 + e^{-b_{t,h}}},
$$

with $\operatorname{softplus}(x) = \log(1 + e^{x})$. $g$ is the decay and $\beta$ the
write strength.

- `a`, `b`: float32, shape `(T, H)`, one row per token.
- `A_log`, `dt_bias`: float32, shape `(H,)`, parameters of the layer.
- Returns `(g, beta)`, both float32 of shape `(T, H)`.

**`gdn_step`** processes one token for all heads. With the head index dropped, and
$g$, $\beta$ the two scalars of that head, it performs three steps in this order:

$$
\begin{aligned}
\text{decay:} \quad & \bar S = g\,S \\
\text{write:} \quad & S' = \bar S + \beta\,(v - \bar S\,k)\,k^\top \\
\text{read:}  \quad & y = S'\,q
\end{aligned}
$$

The correction $v - \bar S k$ is computed from the decayed memory, and the output is
read from the memory that already contains this token's write.

- `S`: float32, shape `(H, Dv, Dk)`.
- `q`, `k`: float32, shape `(H, Dk)`. They are used as given: the step does not
  normalize or scale them (the layer around it does, before it calls the step).
- `v`: float32, shape `(H, Dv)`.
- `g`, `beta`: float32, shape `(H,)`: one row of the gates, or any other values in
  $[0, 1]$.
- Returns `(y, S)`: `y` float32 of shape `(H, Dv)` and the new memory $S'$, float32 of
  shape `(H, Dv, Dk)`.
- The state is the first argument and the second result. The tensors passed in, the
  memory included, are not modified.

Do not call the kit's `gdn_gates_ref`, `gdn_step_ref`, or `ops.gated_delta_layer`.

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

Gates for one token and two heads:

```python
gdn_gates(a=[[0.0, 2.0]], b=[[0.0, -1.0]], A_log=[0.0, -1.0], dt_bias=[0.0, 1.0])
# g    = [[0.5, 0.325817]]
# beta = [[0.5, 0.268941]]
```

Head 0: $\operatorname{softplus}(0) = \ln 2$ and $e^{0} = 1$, so $g = e^{-\ln 2} = 0.5$.
Head 1: $\operatorname{softplus}(3) = 3.0486$ and $e^{-1} = 0.3679$, so
$g = e^{-1.1215} = 0.3258$.

One step for one head with $D_v = D_k = 2$, a unit-length key, $g = 0.5$, and
$\beta = 1$:

```python
S = [[[1.0, 0.0], [0.0, 2.0]]]
q, k, v = [[1.0, 0.0]], [[0.6, 0.8]], [[1.0, -1.0]]

gdn_step(S, q, k, v, g=[0.5], beta=[1.0])
# y = [[0.92, -1.08]]
# S = [[[0.92, 0.56], [-1.08, -0.44]]]
```

![The example step on $S$: decay by $g$, write toward $v$ under $k$, then read with $q$.](assets/delta-step.svg)

Decay gives $\bar S = [[0.5, 0], [0, 1]]$, which returns $\bar S k = (0.3, 0.8)$ for
the key. The correction is $v - \bar S k = (0.7, -1.8)$, and adding its outer product
with $k$ gives $S'$. Now $S' k = (0.552 + 0.448,\; -0.648 - 0.352) = (1, -1) = v$: the
memory returns exactly the new value under this key, whatever it held before. The
output is the first column of $S'$ because $q = (1, 0)$.

With $\beta = 0.5$ the same call moves halfway: `y = [[0.71, -0.54]]` and
`S = [[[0.71, 0.28], [-0.54, 0.28]]]`, for which $S' k = (0.65, -0.1)$, the midpoint of
$(0.3, 0.8)$ and $v$.

## Constraints

- $1 \le T \le 8$, $1 \le H \le 4$, $2 \le D_k \le 16$, $2 \le D_v \le 8$.
- In `gdn_gates`, $|a + \texttt{dt\_bias}| \le 21$ and $|b| \le 20$, so
  $\log(1 + e^{x})$ does not overflow, and $0.01 \le e^{A_{\log}} \le 16$. In exact
  arithmetic $0 < g < 1$; in float32 it may round to 0 or to 1.
- In `gdn_step`, $0 \le g \le 1$ and $0 \le \beta \le 1$; $g = 0$, $g = 1$,
  $\beta = 0$, and $\beta = 1$ all occur. Keys and queries have any length; the memory
  may be all zeros.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`, and for one sequence of six tokens with the outputs of
  the delta rule in Hugging Face Transformers, to the same tolerance.
- With $\beta = 1$ and a unit-length key, $S' k$ must equal $v$ for every $g$, and with
  $g = 1$ a write must leave $S k'$ unchanged for every $k'$ orthogonal to $k$ (same
  tolerance).
- No Python loop over heads is needed.

## Why this matters

Eighteen of the 24 layers of Qwen3.5-0.8B mix tokens with this recurrence instead of
attention, and the memory $S$ is all that such a layer keeps of the prefix: its size
does not grow with the context. Each wrong order of the three steps is a different
model. Reading before the write hides a token from itself, decaying after the write
fades every value before its first use, and a write without the correction is the
additive memory of plain linear attention, which a repeated key saturates. None of
them raises an error, and all of them drift further from the reference with every
token.
