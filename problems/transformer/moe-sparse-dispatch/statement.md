Implement the mixture-of-experts layer the way an inference engine runs it: group the
tokens by the experts they were routed to, run each expert **once** on **only its own
tokens**, and scatter the results back into token order. The routing rule is the one
of `moe-router-forward`; this time the work is counted, and a dense evaluation that
runs every expert on every token is rejected.

![This problem is the MLP slot of a block, run so that each expert sees only its own tokens.](assets/map.svg)

## Specification

```python
def moe_sparse(x, w_router, experts, k): ...
```

- `x`: float32 tensor of shape `(N, D)`, one row per token (batch and positions are
  already flattened).
- `w_router`: float32 tensor of shape `(E, D)`, in `(out, in)` layout.
- `experts`: a [`CountingExperts`](../../../python/src/learnai/kit/moe.py) object holding $E$ SwiGLU experts.
  `experts(e, rows)` runs expert `e` (a Python int) on a float32 tensor `rows` of
  shape `(n, D)` and returns a tensor of shape `(n, D)`, row for row.
  `experts.n_experts` is $E$. This call is the only way to evaluate an expert, and
  it counts: one `expert_calls` per call and `n` `expert_rows`. A call with `n = 0`
  is allowed and still counts as a call.
- `k`: a Python int with $1 \le k \le E$.
- Returns `y`: float32 tensor of shape `(N, D)`.

**Routing**, for each token $x_t$ independently, exactly as in `moe-router-forward`:

1. Router logits $r = W_{\text{router}}\, x_t \in \mathbb{R}^E$ and probabilities
   $p = \operatorname{softmax}(r)$.
2. $S_t$ is the set of the $k$ experts with the largest logits. Experts with equal
   logits are ranked by index, lowest first.
3. Weights $w_{t,e} = p_e / \sum_{j \in S_t} p_j$ for $e \in S_t$.

**Output**, with $f_e$ the function computed by `experts(e, ·)` on one row:

$$
y_t = \sum_{e \in S_t} w_{t,e}\, f_e(x_t).
$$

**Dispatch.** Let $T_e = \{\, t : e \in S_t \,\}$ be the tokens routed to expert $e$,
and call an expert *active* if $T_e$ is not empty. The solution must

- call each active expert **exactly once**, with the rows $x_t$, $t \in T_e$, and
  nothing else (in any row order), and
- **not call** an expert that no token selected, not even with zero rows.

In counters: `expert_rows` equals $k \cdot N$ exactly, and `expert_calls` equals the
number of active experts.

The tokens of $T_e$ are in general scattered over `x`, not adjacent. Row $j$ of an
expert's output belongs to the token whose row was $j$-th in the input of that call,
and has to be added, with that token's weight for this expert, to that token's row of
`y`.

You may use your framework's `softmax` and sorting functions, and plain Python or
NumPy for the bookkeeping of indices and group sizes.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(N, D)` | float32 |
| `w_router` | `(E, D)` | float32 |
| `rows` passed to `experts(e, rows)` | `(n, D)` | float32 |
| result of `experts(e, rows)` | `(n, D)` | float32 |
| `k` | Python int, $1 \le k \le E$ | |
| result | `(N, D)` | float32 |

## Example

Five tokens with $D = 2$, five experts, $k = 2$. The router is chosen so that the
logits of a token $[a, b]$ are $[a,\ b,\ -a,\ -b,\ 0]$:

```python
x = [[3.0, 1.0], [-1.0, 2.0], [2.0, -3.0], [1.0, 3.0], [-3.0, 1.0]]
w_router = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
experts = CountingExperts(fw, seed=1, n_experts=5, d_model=2, d_hidden=4)

y = moe_sparse(x, w_router, experts, 2)
sorted(experts.calls)  # [(0, 3), (1, 4), (2, 2), (3, 1)]   (expert, rows)
```

| token | router logits | selected, best first | weights |
|---|---|---|---|
| 0: $[3, 1]$ | $[3, 1, -3, -1, 0]$ | 0, 1 | 0.8808, 0.1192 |
| 1: $[-1, 2]$ | $[-1, 2, 1, -2, 0]$ | 1, 2 | 0.7311, 0.2689 |
| 2: $[2, -3]$ | $[2, -3, -2, 3, 0]$ | 3, 0 | 0.7311, 0.2689 |
| 3: $[1, 3]$ | $[1, 3, -1, -3, 0]$ | 1, 0 | 0.8808, 0.1192 |
| 4: $[-3, 1]$ | $[-3, 1, 3, -1, 0]$ | 2, 1 | 0.8808, 0.1192 |

Grouped by expert:

| expert | tokens $T_e$ | call |
|---|---|---|
| 0 | 0, 2, 3 | `experts(0, x[[0, 2, 3]])` |
| 1 | 0, 1, 3, 4 | `experts(1, x[[0, 1, 3, 4]])` |
| 2 | 1, 4 | `experts(2, x[[1, 4]])` |
| 3 | 2 | `experts(3, x[[2]])` |
| 4 | none | no call |

![A filled cell is a token routed to an expert, with its weight; each row is one call.](assets/dispatch.svg)

Four calls and $3 + 4 + 2 + 1 = 10 = k \cdot N$ rows. The dense evaluation makes five
calls with 25 rows. Expert 0's tokens are 0, 2, and 3: the three rows of its output go
to rows 0, 2, and 3 of `y`, with weights 0.8808, 0.2689, and 0.1192. Then
$y_0 = 0.8808\, f_0(x_0) + 0.1192\, f_1(x_0)$, and so on.

## Constraints

- $1 \le N \le 256$, $2 \le D \le 32$, $1 \le E \le 8$, $1 \le k \le E$.
- The result is compared with the dense evaluation of the same layer to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-5`. An expert's output for a
  row does not depend on which other rows are in the call, except in the last bits.
- **Cost, checked in every case:** `expert_rows` is exactly $k \cdot N$ and
  `expert_calls` is at most the number of active experts (so, with the right output,
  exactly that number).
- In every test, the $k$-th and $(k+1)$-th largest router logits of a token are
  either exactly equal (inputs made of small integers, for which float32 arithmetic
  is exact) or differ by more than $10^{-3}$, so rounding cannot change the routing.
- A Python loop over experts is expected. No loop over tokens is needed.

## Why this matters

Sparsity is the point of a mixture of experts: Mixtral 8x7B runs 2 of its 8 expert
MLPs per token and Qwen3-30B-A3B 8 of 128, so the dense evaluation costs 4 and 16
times the MLP compute the model was designed around. Engines get the sparse cost with
this gather, run, and scatter pattern, and its failure modes are the ones tested
here. One call per token gives the right arithmetic and destroys throughput, because
a GPU wants a few large matrix products, not thousands of single-row ones. Calling
idle experts with empty batches costs a kernel launch each, which adds up at 128
experts and small batches. And a scatter that assumes an expert's tokens are adjacent
passes every test with one sequence of similar tokens, then mixes hidden states
between unrelated tokens under real batching. It is also the layout that expert
parallelism builds on: once tokens are grouped by expert, each group can be sent to
the device that holds that expert.
