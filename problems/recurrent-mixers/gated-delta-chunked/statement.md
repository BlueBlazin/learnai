Implement the gated delta rule over a full sequence while accessing the recurrent memory only once per chunk instead of once per token. Inside each chunk, every token's write depends on the earlier writes in that chunk, so you must resolve those within-chunk dependencies before touching the memory.

## Specification

```python
def gdn_chunked(q, k, v, log_g, beta, memory, chunk_size): ...  # -> (Y, memory)
```

The sequence has $T$ tokens and $H$ heads, with key dimension $D_k$ and value dimension $D_v$. Each head maintains a state matrix $S \in \mathbb{R}^{D_v \times D_k}$ stored inside `memory`. For each token $t = 0, \dots, T - 1$ and each head, the inputs provide a query $q_t \in \mathbb{R}^{D_k}$, a key $k_t \in \mathbb{R}^{D_k}$, a value $v_t \in \mathbb{R}^{D_v}$, a log-decay $\texttt{log\_g}_t \le 0$ (with decay $g_t = e^{\texttt{log\_g}_t}$), and a write rate $\beta_t \in [0, 1]$.

**Target computation.** Starting from the initial matrix $S$ in `memory`, applying the token-by-token rule of `gated-delta-step` for $t = 0, \dots, T - 1$ computes

$$
\bar S = g_t\,S, \qquad S = \bar S + \beta_t\,(v_t - \bar S k_t)\,k_t^\top, \qquad y_t = S\,q_t .
$$

Your function must return `(Y, memory)`, where `Y` collects every output $y_t$ and the returned `memory` holds the final state $S$ after token $T - 1$. The output and final state must not depend on `chunk_size`. Do not modify any input tensors.

**The memory interface.** The `memory` argument is an opaque object that exposes only two operations and one attribute:

- `memory.read(rows)` takes a tensor `rows` of shape `(H, R, Dk)` (for $R$ rows per head) and returns `rows @ S^T` of shape `(H, R, Dv)`.
- `memory.update(decay, values, keys)` takes `decay` of shape `(H,)`, `values` of shape `(H, R, Dv)`, and `keys` of shape `(H, R, Dk)`. It leaves the current memory unchanged and returns a new memory holding $\texttt{decay} \cdot S + \texttt{values}^\top \texttt{keys}$ per head.
- `memory.max_rows` is the maximum number of rows $R$ allowed in a single call ($1 \le R \le \texttt{memory.max\_rows}$). Every test sets `max_rows` to `2 * chunk_size`.

Each call to `read` or `update` counts as one access regardless of $R$. Let $n = \lceil T / \texttt{chunk\_size} \rceil$ be the number of chunks. You may make **at most $n$ calls to `read` and at most $n$ calls to `update`** (one of each per chunk). The second return value must be the memory returned by your last `update` call.

**Chunked formulation.** Partition the sequence into consecutive chunks of `chunk_size` tokens. The final chunk may be shorter than `chunk_size`, and `chunk_size` (a Python `int` $\ge 1$) may be larger than $T$.

Consider one head and one chunk of $C$ tokens. Stack the chunk's queries, keys, and values as rows of $Q, K \in \mathbb{R}^{C \times D_k}$ and $V \in \mathbb{R}^{C \times D_v}$, and let $S_0$ be the memory before the chunk. Define the cumulative log-decay $c_i = \sum_{j \le i} \texttt{log\_g}_j$ and the pairwise decay $G_{ij} = e^{c_i - c_j}$ for $i \ge j$.

1. **Triangular system.** Define the strictly lower-triangular matrix $A$ by $A_{ij} = \beta_i\,(k_i \cdot k_j)\,G_{ij}$ for $j < i$, and $A_{ij} = 0$ otherwise. Solve the two unit-lower-triangular linear systems

$$
(I + A)\,U = \operatorname{diag}(\beta)\,V, \qquad
(I + A)\,W = \operatorname{diag}(\beta\,e^{c})\,K .
$$

2. **Chunk writes.** Compute $V' = U - W S_0^\top$ of shape $(C, D_v)$. Row $i$ of $V'$ is the exact correction $\beta_i\,(v_i - \bar S k_i)$ written at position $i$ by the step rule.
3. **Chunk reads.** Define the lower-triangular read matrix $R$ by $R_{ij} = (q_i \cdot k_j)\,G_{ij}$ for $j \le i$ (including the diagonal), and $R_{ij} = 0$ otherwise. Then

$$
Y = \operatorname{diag}(e^{c})\,Q\,S_0^\top + R\,V' .
$$

4. **Memory update.** Advance the state to the end of the chunk via

$$
S_1 = e^{c_{C-1}}\,S_0 + V'^\top \operatorname{diag}(e^{c_{C-1} - c})\,K .
$$

Because $A$, $U$, $W$, and $R$ do not depend on $S_0$, they can be computed for all chunks upfront. Steps 2 and 3 depend on $S_0$ only through $W S_0^\top$ and $\operatorname{diag}(e^{c})\,Q\,S_0^\top$. Stacking those $2C$ rows lets you fetch both in a single `memory.read`, and step 4 is a single `memory.update`.

> [!WARNING]
> When $i < j$, the exponent $c_i - c_j$ is positive and can overflow float32 under fast decay. Never compute $e^{c_i - c_j}$ for $i < j$ and multiply by zero afterward, because $\infty \cdot 0$ produces NaN. Mask the exponent before calling `exp`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, `k` | `(T, H, Dk)` | float32 |
| `v`, `Y` | `(T, H, Dv)` | float32 |
| `log_g`, `beta` | `(T, H)` | float32 |
| matrices inside `memory` | `(H, Dv, Dk)` | float32 |

## Example

Consider three tokens, one head, $D_k = 2$, $D_v = 1$, and `chunk_size = 2`:

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

**First chunk (tokens 0 and 1).** The cumulative decays are $e^{c} = (0.5, 0.25)$ and the pairwise decay is $G_{10} = 0.5$.

- **System.** $A_{10} = \beta_1 (k_1 \cdot k_0)\,G_{10} = 0.5 \cdot 0.6 \cdot 0.5 = 0.15$. Forward substitution gives $U = (1,\; 1 - 0.15 \cdot 1) = (1, 0.85)$. From the right-hand rows $(0.5, 0)$ and $(0.075, 0.1)$, forward substitution gives $W = [(0.5, 0),\ (0, 0.1)]$.
- **Writes.** $V' = (1 - 0.5 \cdot 2,\; 0.85 - 0.1 \cdot (-2)) = (0, 1.05)$. This matches the step rule, where token 0 finds its target value already stored and token 1 writes $0.5\,(2 + 0.1)$.
- **Reads.** $y_0 = 0.5 \cdot 2 + 1 \cdot 0 = 1$ and $y_1 = 0.25 \cdot (-2) + 0 \cdot 0.5 \cdot 0 + 0.8 \cdot 1.05 = 0.34$.
- **Memory.** $S_1 = 0.25\,(2, -2) + 0 \cdot 0.5\,(1, 0) + 1.05\,(0.6, 0.8) = (1.13, 0.34)$.

**Second chunk (token 2).** With $g_2 = 1$ and $\beta_2 = 1$, token 2 writes $-1 - 0.34 = -1.34$ along $(0, 1)$. That yields $S = (1.13, -1)$ and $y_2 = 0.6 \cdot 1.13 - 0.8 = -0.122$.

## Constraints

- $1 \le T \le 40$, $1 \le H \le 4$, $2 \le D_k \le 64$, $1 \le D_v \le 16$, and $1 \le \texttt{chunk\_size} \le 16$.
- Keys have norm 1 and queries have norm $D_k^{-1/2}$ (except in the worked example). Log-decays satisfy $-30 \le \texttt{log\_g} \le 0$, and write rates satisfy $0 \le \beta \le 1$ including both endpoints.
- `Y` and the returned memory are checked against the token-by-token rule in float64, and on two test cases against `torch_chunk_gated_delta_rule` in Hugging Face Transformers, with relative tolerance `1e-3` and absolute tolerance `1e-4`.
- **Access budget.** Every test checks that you make at most $n$ calls to `read` and at most $n$ calls to `update`. A token-by-token loop requires one read per token and fails whenever `chunk_size > 1`. In one test, keys have 64 channels while the allowed reads return at most 48 numbers per value channel, preventing you from copying the full state out of `memory`.
- Any valid way of solving the triangular systems passes, including a library solver, forward substitution across chunk positions, or a row-by-row loop inside each chunk. In MLX 0.32, `mx.linalg.solve_triangular` requires `stream=mx.cpu`.
- Do not access `memory`'s private attributes, `peek`, or `counters`.
- In JAX, every distinct array shape triggers a compilation. Forward substitution that builds the solution row by row encounters 16 shapes, so the first submission can take around 10 seconds.

## Why this matters

A recurrent layer looks inherently sequential, as if prefilling a prompt must step through one token at a time while waiting for the previous memory state. The chunked form breaks that bottleneck. All pairwise interactions inside a chunk become batched matrix arithmetic that never touches the memory, and the memory is read and updated only once per chunk. The Qwen3.5 implementation runs this chunked algorithm with chunk size 64 for every multi-token call, falling back to the per-token rule only when decoding a single cached token.
