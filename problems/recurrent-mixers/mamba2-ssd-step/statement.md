Implement both the single-token recurrent step and the full-sequence matrix form of the Mamba-2 state-space layer used in Nemotron, taking the step-size clamp as an explicit argument. Both forms must agree because the recurrent layer is equivalent to attention with a computed decay mask.

## Specification

```python
def ssd_step(h, x, dt_raw, B, C, A_log, D, dt_bias, dt_limit): ...  # -> (y, h)
def ssd_matrix(dt, B, C, A_log): ...  # -> M
```

The layer has $H$ heads, each with $P$ channels and a state of size $N$ per channel. Head $i$ owns a state matrix $h_i \in \mathbb{R}^{P \times N}$. The input and output projection vectors $B$ and $C$ are shared across $G$ groups, where each group serves $H / G$ consecutive heads. Head $i$ uses group $\gamma(i) = \lfloor i \,/\, (H / G) \rfloor$.

**`ssd_step`** advances the state by one token. Its inputs are the current state `h`, the token's per-head input `x`, the raw per-head step size `dt_raw`, the per-group vectors `B` and `C`, the per-head layer parameters `A_log`, `D`, and `dt_bias`, and `dt_limit`, a tuple `(low, high)` of Python floats with $0 \le \texttt{low} < \texttt{high}$ (`high` may be `float("inf")`). For each head $i$, compute in order:

$$
\begin{aligned}
\Delta_i &= \operatorname{clip}\bigl(\operatorname{softplus}(\texttt{dt\_raw}_i + \texttt{dt\_bias}_i),\ \texttt{dt\_limit}[0],\ \texttt{dt\_limit}[1]\bigr) \\
h_i' &= e^{-\Delta_i\, e^{A_{\log,i}}}\; h_i \;+\; \Delta_i\; x_i\, B_{\gamma(i)}^\top \\
y_i &= h_i'\, C_{\gamma(i)} + D_i\, x_i
\end{aligned}
$$

Here $\operatorname{softplus}(u) = \log(1 + e^{u})$, $x_i \in \mathbb{R}^{P}$, and $x_i B_{\gamma(i)}^\top$ is the $P \times N$ outer product. Notice that the output $y_i$ is read from the updated state $h_i'$.

Return `(y, h')`, with the state passed as the first argument and returned as the second result. Do not modify any input tensors, including `h`.

**`ssd_matrix`** builds the $T \times T$ mixing matrix for $T$ tokens at once. Here `dt[t, i]` is the step size $\Delta_{t,i} > 0$ after softplus and clamping, and `B[t]`, `C[t]` are the group vectors at token $t$. For each head $i$, return the lower-triangular matrix

$$
M_{i,t,s} =
\begin{cases}
\bigl(C_{t,\gamma(i)} \cdot B_{s,\gamma(i)}\bigr)\;
\exp\!\Bigl(-e^{A_{\log,i}} \displaystyle\sum_{r = s+1}^{t} \Delta_{r,i}\Bigr)\; \Delta_{s,i} & s \le t, \\[2ex]
0 & s > t .
\end{cases}
$$

The sum is empty when $s = t$, so the exponential factor on the diagonal is $1$. Every entry above the diagonal ($s > t$) must be exactly zero. Applied to an input sequence $X$ of shape `(T, H, P)`, this matrix reproduces `ssd_step` run token by token from a zero initial state:

$$
y_{t,i} = \sum_{s} M_{i,t,s}\, x_{s,i} + D_i\, x_{t,i} .
$$

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

Consider one head with $P = 1$, $N = 2$, and limits `(0.001, inf)`:

```python
h, x = [[[1.0, 2.0]]], [[2.0]]
B, C = [[1.0, -1.0]], [[0.5, 0.25]]
A_log, D, dt_bias = [0.0], [3.0], [0.0]

ssd_step(h, x, [0.0], B, C, A_log, D, dt_bias, (0.001, float("inf")))
# y = [[6.846574]],  h = [[[1.886294, -0.386294]]]
ssd_step(h, x, [-10.0], B, C, A_log, D, dt_bias, (0.001, float("inf")))
# y = [[6.999500]],  h = [[[1.001000, 1.996002]]]
```

In the first call, $\Delta = \operatorname{softplus}(0) = \ln 2 \approx 0.6931$. The state decays by $e^{-0.6931} = 0.5$ to $(0.5, 1)$, the input adds $0.6931 \cdot 2 \cdot (1, -1)$, and $y = 1.8863 \cdot 0.5 - 0.3863 \cdot 0.25 + 3 \cdot 2$. In the second call, $\operatorname{softplus}(-10) \approx 0.0000454$ falls below the lower bound, so the clamp sets $\Delta = 0.001$.

Now consider `ssd_matrix` for three tokens of one head with $e^{A_{\log}} = \ln 2$, so a step of size $1$ halves the state:

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

For entry $(2, 0)$, $M_{2,0} = (C_2 \cdot B_0) \cdot 2^{-(1 + 2)} \cdot \Delta_0 = 1 \cdot 0.125 \cdot 0.5 = 0.0625$. Token 0's input decays through the steps of tokens 1 and 2, not through its own step. Similarly, $M_{2,1} = -0.5 \cdot 2^{-2} \cdot 1 = -0.125$, and the diagonal entries carry no decay.

## Constraints

- $1 \le H \le 4$, $G$ divides $H$, $1 \le P \le 4$, $2 \le N \le 5$, and $1 \le T \le 9$.
- `dt_limit` is one of `(0.001, inf)`, `(0.001, 100.0)`, or `(0.0, inf)`. Both bounds are tested, with $\texttt{dt\_raw} + \texttt{dt\_bias}$ ranging from $-20$ to $150$.
- $0.3 \le e^{A_{\log}} \le 4$, and in `ssd_matrix`, $e^{A_{\log,i}} \sum_t \Delta_{t,i} \le 60$, so no exponential overflows.
- Outputs are checked against the reference, and on two sequences against the Mamba-2 functions in Hugging Face Transformers, with relative tolerance `1e-4` and absolute tolerance `1e-5`. Under limits `(0.0, inf)`, the state change $h' - h$ is also checked with relative tolerance `2e-2` and absolute tolerance `1e-4`.
- Applying `M X + D X` must match looping `ssd_step` from a zero state within the same tolerances.
- No Python loops over heads, groups, or positions are needed.

## Why this matters

Inference engines use the matrix form in chunks for prompt prefill and the step form for token-by-token decoding, so two separate code paths must implement the exact same layer. In the reference Nemotron implementation, the two paths disagree on clamping. The chunked path clamps the step size at `0.001`, while the single-token path does not. A port that copies the wrong rule from one path into the other produces a model whose outputs change depending on whether a token was prefilled or decoded.
