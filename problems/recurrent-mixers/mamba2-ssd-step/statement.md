Implement one token of the Mamba-2 state-space layer that Nemotron uses, with the
step-size clamp as an argument, and the matrix that the same layer applies to a whole
sequence. The two must agree: the recurrent layer is attention with a mask that is
computed instead of learned.

## Specification

```python
def ssd_step(h, x, dt_raw, B, C, A_log, D, dt_bias, dt_limit): ...  # -> (y, h)
def ssd_matrix(dt, B, C, A_log): ...  # -> M
```

There are $H$ heads of width $P$ and a state of size $N$ per channel: head $i$ owns
$h_i \in \mathbb{R}^{P \times N}$. The vectors $B$ and $C$ come in $G$ groups, each
serving $H / G$ consecutive heads: head $i$ uses the vectors of group
$\gamma(i) = \lfloor i \,/\, (H / G) \rfloor$.

**`ssd_step`** processes one token. For each head $i$, in this order:

$$
\begin{aligned}
\Delta_i &= \operatorname{clip}\bigl(\operatorname{softplus}(\texttt{dt\_raw}_i + \texttt{dt\_bias}_i),\ \texttt{dt\_limit}[0],\ \texttt{dt\_limit}[1]\bigr) \\
h_i' &= e^{-\Delta_i\, e^{A_{\log,i}}}\; h_i \;+\; \Delta_i\; x_i\, B_{\gamma(i)}^\top \\
y_i &= h_i'\, C_{\gamma(i)} + D_i\, x_i
\end{aligned}
$$

with $\operatorname{softplus}(u) = \log(1 + e^{u})$, $x_i \in \mathbb{R}^{P}$, and
$x_i B^\top$ the $P \times N$ outer product. The output is read from the updated
state.

- `h`: float32, shape `(H, P, N)`. `x`: float32, shape `(H, P)`.
- `dt_raw`: float32, shape `(H,)`, the token's raw step size per head.
- `B`, `C`: float32, shape `(G, N)`, the token's input and output vectors per group.
- `A_log`, `D`, `dt_bias`: float32, shape `(H,)`, parameters of the layer.
- `dt_limit`: a tuple `(low, high)` of Python floats, $0 \le \texttt{low} <
  \texttt{high}$; `high` may be `float("inf")`.
- Returns `(y, h)`: float32 of shape `(H, P)` and the new state, float32 of shape
  `(H, P, N)`. The state is the first argument and the second result; the tensors
  passed in, the state included, are not modified.

**`ssd_matrix`** describes $T$ tokens at once. `dt[t, i]` is the step size
$\Delta_{t,i}$ after softplus and clamp, and `B[t]`, `C[t]` are the vectors of token
$t$. It returns, for each head, the $T \times T$ matrix

$$
M_{i,t,s} =
\begin{cases}
\bigl(C_{t,\gamma(i)} \cdot B_{s,\gamma(i)}\bigr)\;
\exp\!\Bigl(-e^{A_{\log,i}} \displaystyle\sum_{r = s+1}^{t} \Delta_{r,i}\Bigr)\; \Delta_{s,i} & s \le t, \\[2ex]
0 & s > t .
\end{cases}
$$

The sum is empty for $s = t$. Applied to the inputs $X$ of shape `(T, H, P)`, it gives
what `ssd_step` returns token by token from a zero state:
$y_{t,i} = \sum_{s} M_{i,t,s}\, x_{s,i} + D_i\, x_{t,i}$.

- `dt`: float32, shape `(T, H)`, every entry $> 0$.
- `B`, `C`: float32, shape `(T, G, N)`. `A_log`: float32, shape `(H,)`.
- Returns float32 of shape `(H, T, T)`, exactly zero above the diagonal.

## Shapes

| name | shape | dtype |
|---|---|---|
| `h` (argument and result) | `(H, P, N)` | float32 |
| `x`, `y` | `(H, P)` | float32 |
| `dt_raw`, `A_log`, `D`, `dt_bias` | `(H,)` | float32 |
| `B`, `C` in `ssd_step` | `(G, N)` | float32 |
| `dt` | `(T, H)` | float32 |
| `B`, `C` in `ssd_matrix` | `(T, G, N)` | float32 |
| `M` | `(H, T, T)` | float32 |

## Example

One head, $P = 1$, $N = 2$, limits `(0.001, inf)`:

```python
h, x = [[[1.0, 2.0]]], [[2.0]]
B, C = [[1.0, -1.0]], [[0.5, 0.25]]
A_log, D, dt_bias = [0.0], [3.0], [0.0]

ssd_step(h, x, [0.0], B, C, A_log, D, dt_bias, (0.001, float("inf")))
# y = [[6.846574]],  h = [[[1.886294, -0.386294]]]
ssd_step(h, x, [-10.0], B, C, A_log, D, dt_bias, (0.001, float("inf")))
# y = [[6.999500]],  h = [[[1.001000, 1.996002]]]
```

First call: $\Delta = \operatorname{softplus}(0) = \ln 2 = 0.6931$, the state decays by
$e^{-0.6931} = 0.5$ to $(0.5, 1)$, the input adds $0.6931 \cdot 2 \cdot (1, -1)$, and
$y = 1.8863 \cdot 0.5 - 0.3863 \cdot 0.25 + 3 \cdot 2$. Second call:
$\operatorname{softplus}(-10) = 0.0000454$ is below the limit, so $\Delta = 0.001$.

A matrix for three tokens of one head with $e^{A_{\log}} = \ln 2$, so that a step of
size 1 halves the state:

```python
dt = [[0.5], [1.0], [2.0]]
B = [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]
C = [[[1.0, 1.0]], [[2.0, 1.0]], [[1.0, -0.5]]]

ssd_matrix(dt, B, C, A_log=[log(log(2.0))])
# [[[0.5,     0.0,   0.0],
#   [0.5,     1.0,   0.0],
#   [0.0625, -0.125, 1.0]]]
```

![The matrix $M$ of the three-token example. Each cell fades with the decay from input token $s$ to output token $t$; above the diagonal it is zero.](assets/ssd-matrix.svg)

$M_{2,0} = (C_2 \cdot B_0) \cdot 2^{-(1 + 2)} \cdot \Delta_0 = 1 \cdot 0.125 \cdot 0.5$: the
input of token 0 has decayed through the steps of tokens 1 and 2, not through its own.
$M_{2,1} = -0.5 \cdot 2^{-2} \cdot 1$, and the diagonal carries no decay.

## Constraints

- $1 \le H \le 4$, $G$ divides $H$, $1 \le P \le 4$, $2 \le N \le 5$, $1 \le T \le 9$.
- `dt_limit` is one of `(0.001, inf)`, `(0.001, 100.0)`, `(0.0, inf)`. Both limits are
  exercised: $\texttt{dt\_raw} + \texttt{dt\_bias}$ ranges from $-20$ to $150$.
- $0.3 \le e^{A_{\log}} \le 4$, and in `ssd_matrix`
  $e^{A_{\log,i}} \sum_t \Delta_{t,i} \le 60$, so no exponential of that sum overflows.
- Results are compared with the reference, and for two sequences with the outputs of
  the Mamba-2 functions in Hugging Face Transformers: relative tolerance `1e-4`,
  absolute `1e-5`. With limits `(0.0, inf)` the change of the state,
  $h' - h$, is also compared, to a relative tolerance of `2e-2` and an absolute one of
  `1e-4`.
- `M X + D X` must equal the loop of `ssd_step` from a zero state (same tolerance).
- No Python loop over heads, groups, or positions is needed.

## Why this matters

Decoding calls the step and prefill uses the matrix form in chunks, so the two are
separate code paths that must describe one layer. In the reference implementation of
Nemotron they differ in the clamp: the chunked path cuts the step size at 0.001 and
the one-token path does not. A port that copies one path's rule into the other, or
reads the wrong one as the specification, produces a model whose answers depend on
whether a token was prefilled or decoded.
