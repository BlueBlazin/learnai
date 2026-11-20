Compute the gated delta rule for a whole sequence while touching the memory once per
chunk instead of once per token. Inside a chunk every token's write depends on the
writes before it, and that dependence has to be resolved without the memory.

## Specification

```python
def gdn_chunked(q, k, v, log_g, beta, memory, chunk_size): ...  # -> (Y, memory)
```

**What it returns.** For each head, starting from its matrix $S$ in `memory`, shape
$(D_v, D_k)$, and for
$t = 0, \dots, T - 1$ in order, the rule of `gated-delta-step` with
$g_t = e^{\texttt{log\_g}_t}$:

$$
\bar S = g_t\,S, \qquad S = \bar S + \beta_t\,(v_t - \bar S k_t)\,k_t^\top, \qquad y_t = S\,q_t .
$$

`Y` holds every $y_t$, and the returned memory holds $S$ after the last token.
`chunk_size` never changes the result.

**The memory is an object**, and these two calls are the only access to it:

- `memory.read(rows)`: `rows` of shape `(H, R, Dk)`; returns `rows @ S^T`, shape
  `(H, R, Dv)`: what each head's memory returns for each of its $R$ rows.
- `memory.update(decay, values, keys)`: `decay` of shape `(H,)`, `values` `(H, R, Dv)`,
  `keys` `(H, R, Dk)`; returns a **new** memory holding
  $\texttt{decay} \cdot S + \texttt{values}^\top \texttt{keys}$ per head. The memory it
  is called on is unchanged.

One call takes $1 \le R \le$ `memory.max_rows` rows, and `max_rows` is
`2 * chunk_size` in every test. Each call counts once, whatever $R$ is. With
$n = \lceil T / \texttt{chunk\_size} \rceil$ chunks, **at most $n$ reads and at
most $n$ updates** are allowed: one of each per chunk. The second result must be the
memory returned by your last `update`.

**How.** Split the sequence into consecutive chunks of `chunk_size` tokens (the last
one may be shorter). For one head and one chunk of $C$ tokens, stack its rows into
$Q, K \in \mathbb{R}^{C \times D_k}$ and $V \in \mathbb{R}^{C \times D_v}$, and let $S_0$
be the memory before the chunk. With the cumulative log-decay
$c_i = \sum_{j \le i} \texttt{log\_g}_j$ and $G_{ij} = e^{c_i - c_j}$ for $i \ge j$:

1. **The system.** $A_{ij} = \beta_i\,(k_i \cdot k_j)\,G_{ij}$ for $j < i$ and
   $A_{ij} = 0$ otherwise. Solve the two unit-lower-triangular systems

$$
(I + A)\,U = \operatorname{diag}(\beta)\,V, \qquad
(I + A)\,W = \operatorname{diag}(\beta\,e^{c})\,K .
$$

2. **The writes.** $V' = U - W S_0^\top$, shape $(C, D_v)$. Row $i$ is the correction
   $\beta_i (v_i - \bar S k_i)$ that the step rule writes at position $i$.
3. **The reads.** With $R_{ij} = (q_i \cdot k_j)\,G_{ij}$ for $j \le i$ (diagonal
   included) and $R_{ij} = 0$ otherwise:

$$
Y = \operatorname{diag}(e^{c})\,Q\,S_0^\top + R\,V' .
$$

4. **The memory after the chunk.**
   $S_1 = e^{c_{C-1}} S_0 + V'^\top \operatorname{diag}(e^{c_{C-1} - c})\,K$.

$A$, $U$, $W$, and $R$ do not depend on $S_0$. Steps 2 and 3 need $S_0$ only through
$W S_0^\top$ and $\operatorname{diag}(e^{c}) Q\,S_0^\top$, which is one `read` of $2C$
stacked rows, and step 4 is one `update`.

- `q`, `k`: float32, shape `(T, H, Dk)`; `v`: float32, shape `(T, H, Dv)`. Used as given.
- `log_g`: float32, shape `(T, H)`, every entry $\le 0$. `beta`: float32, shape
  `(T, H)`, in $[0, 1]$.
- `memory`: the object above, holding float32 matrices of shape `(H, Dv, Dk)`.
- `chunk_size`: a Python int $\ge 1$. It may exceed $T$, and $T$ need not be a multiple
  of it.
- Returns `(Y, memory)`: float32 of shape `(T, H, Dv)` and the updated memory.
- The tensors passed in are not modified.

> [!WARNING]
> For $i < j$ the number $e^{c_i - c_j}$ is the exponential of a positive quantity and
> overflows float32 when the decay is fast. Such entries must never be computed and
> then multiplied by zero: $\infty \cdot 0$ is NaN.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, `k` | `(T, H, Dk)` | float32 |
| `v`, `Y` | `(T, H, Dv)` | float32 |
| `log_g`, `beta` | `(T, H)` | float32 |
| the matrices inside `memory` | `(H, Dv, Dk)` | float32 |

## Example

Three tokens, one head, $D_k = 2$, $D_v = 1$, `chunk_size = 2`:

```python
q = [[[1.0, 0.0]], [[0.0, 1.0]], [[0.6, 0.8]]]
k = [[[1.0, 0.0]], [[0.6, 0.8]], [[0.0, 1.0]]]
v = [[[1.0]], [[2.0]], [[-1.0]]]
log_g = log([[0.5], [0.5], [1.0]])
beta = [[1.0], [0.5], [1.0]]
# memory holds S0 = [[[2.0, -2.0]]]

gdn_chunked(q, k, v, log_g, beta, memory, 2)
# Y = [[[1.0]], [[0.34]], [[-0.122]]]
# the returned memory holds [[[1.13, -1.0]]], after 2 reads and 2 updates
```

The first chunk holds tokens 0 and 1: $e^{c} = (0.5, 0.25)$ and $G_{10} = 0.5$.

- System: $A_{10} = \beta_1 (k_1 \cdot k_0)\,G_{10} = 0.5 \cdot 0.6 \cdot 0.5 = 0.15$.
  Forward substitution gives $U = (1,\; 1 - 0.15 \cdot 1) = (1, 0.85)$ and, from the
  right-hand rows $(0.5, 0)$ and $(0.075, 0.1)$, $W = [(0.5, 0),\ (0, 0.1)]$.
- Writes: $V' = (1 - 0.5 \cdot 2,\ \ 0.85 - 0.1 \cdot (-2)) = (0, 1.05)$. The step rule
  agrees: token 0 finds its value already stored, token 1 writes $0.5\,(2 + 0.1)$.
- Reads: $y_0 = 0.5 \cdot 2 + 1 \cdot 0 = 1$ and
  $y_1 = 0.25 \cdot (-2) + 0 \cdot 0.5 \cdot 0 + 0.8 \cdot 1.05 = 0.34$.
- Memory: $S_1 = 0.25\,(2, -2) + 0 \cdot 0.5\,(1, 0) + 1.05\,(0.6, 0.8) = (1.13, 0.34)$.

The second chunk is token 2 alone: it writes $-1 - 0.34 = -1.34$ along $(0, 1)$,
giving $S = (1.13, -1)$ and $y_2 = 0.6 \cdot 1.13 - 0.8 = -0.122$.

## Constraints

- $1 \le T \le 40$, $1 \le H \le 4$, $2 \le D_k \le 64$, $1 \le D_v \le 16$,
  $1 \le \texttt{chunk\_size} \le 16$.
- Keys have length 1 and queries length $D_k^{-1/2}$ (the worked example aside),
  $-30 \le \texttt{log\_g} \le 0$, and $0 \le \beta \le 1$ including both ends.
- `Y` and the returned memory are compared with the token-by-token rule evaluated in
  float64, and for two inputs with the outputs of `torch_chunk_gated_delta_rule` in
  Hugging Face Transformers: relative tolerance `1e-3`, absolute `1e-4`.
- **Cost, checked in every case:** at most $n$ `read` calls and at most $n$ `update`
  calls. A loop over tokens needs one read per token and fails whenever
  `chunk_size` is above 1. In one test the keys have 64 channels while all the reads
  the limit allows return 48 numbers per value channel, so the memory cannot be copied
  out and processed elsewhere.
- How the triangular systems are solved is not judged: a library solver, forward
  substitution over the positions of a chunk, or a row-by-row loop inside each chunk
  all pass. In MLX 0.32, `mx.linalg.solve_triangular` needs `stream=mx.cpu`.
- Do not use the memory's private attributes, its `peek` method, or its `counters`.
- In JAX, each new array shape is compiled once; a forward substitution that builds
  the solution row by row meets 16 shapes, and a first submit can take about 10
  seconds.

## Why this matters

A recurrent layer looks as if it must read a prompt one token at a time, each step
waiting for the memory the previous one left. The chunked form is why it does not:
the pairwise work inside a chunk is batched matrix arithmetic that never sees the
memory, and the memory is read and written once per chunk. The library that ships
Qwen3.5 runs this form for every call with more than one token, with chunks of 64,
and uses the per-token rule only for single cached tokens.
