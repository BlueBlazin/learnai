Implement a sparse mixture-of-experts (MoE) layer the way an inference engine runs it. Group tokens by their assigned experts, run each active expert **once** on **only its own tokens**, and scatter the weighted results back into token order. The routing rule matches `moe-router-forward`, except expert work is counted, so a dense implementation that runs every expert on every token is rejected.

![This problem is the MLP slot of a block, run so that each expert sees only its own tokens.](assets/map.svg)

## Specification

```python
def moe_sparse(x, w_router, experts, k): ...
```

The input `x` holds `N` token vectors of dimension `D`, with batch and sequence axes already flattened. The router weight matrix `w_router` maps each token to `E` expert logits in `(out, in)` layout. The integer `k` ($1 \le k \le E$) is the number of experts selected per token.

The `experts` argument is a [`CountingExperts`](../../../python/src/learnai/kit/moe.py) object holding `E` SwiGLU experts, with `experts.n_experts == E`. Calling `experts(e, rows)` runs expert `e` (a Python `int`) independently on each row of `rows` and returns a tensor of the same shape. This call is the only way to evaluate an expert, and every call updates two counters: `expert_calls` increases by 1 and `expert_rows` increases by `rows.shape[0]`. Calling an expert with `0` rows is valid and still increments `expert_calls`.

**Routing.** For each token $x_t$ independently, follow the routing rule from `moe-router-forward`:

1. Compute router logits $r = W_{\text{router}}\, x_t \in \mathbb{R}^E$ and probabilities $p = \operatorname{softmax}(r)$.
2. Let $S_t$ be the set of $k$ experts with the largest logits. Break ties by expert index, lowest first.
3. Compute normalized routing weights $w_{t,e} = p_e / \sum_{j \in S_t} p_j$ for each $e \in S_t$.

**Output.** Let $f_e$ denote the function computed by `experts(e, ·)` on a single token row. The output row for token $t$ is:

$$
y_t = \sum_{e \in S_t} w_{t,e}\, f_e(x_t).
$$

**Dispatch.** Let $T_e = \{\, t : e \in S_t \,\}$ be the set of tokens routed to expert $e$, and call expert $e$ *active* if $T_e$ is non-empty. Your implementation must satisfy two rules:

- Call each active expert **exactly once**, passing only the rows $x_t$ for $t \in T_e$ (in any row order).
- **Never call** an inactive expert, not even with zero rows.

In terms of the counters, `expert_rows` must equal $k \cdot N$ exactly, and `expert_calls` must equal the number of active experts.

Because the tokens in $T_e$ are generally scattered across `x`, row $j$ of an expert's output corresponds to the $j$-th token passed into that call. Multiply it by that token's weight for expert $e$ and accumulate it into that token's row of `y`.

You may use your framework's `softmax` and sorting functions, as well as plain Python or NumPy for index and group-size bookkeeping.

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

Consider five tokens with $D = 2$, five experts, and $k = 2$. The router matrix is chosen so that a token $[a, b]$ produces logits $[a,\ b,\ -a,\ -b,\ 0]$:

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

Grouping the assignments by expert gives four active experts:

| expert | tokens $T_e$ | call |
|---|---|---|
| 0 | 0, 2, 3 | `experts(0, x[[0, 2, 3]])` |
| 1 | 0, 1, 3, 4 | `experts(1, x[[0, 1, 3, 4]])` |
| 2 | 1, 4 | `experts(2, x[[1, 4]])` |
| 3 | 2 | `experts(3, x[[2]])` |
| 4 | none | no call |

![A filled cell is a token routed to an expert, with its weight; each row is one call.](assets/dispatch.svg)

This uses 4 calls and $3 + 4 + 2 + 1 = 10 = k \cdot N$ rows, compared to 5 calls and 25 rows for a dense evaluation. Expert 0 processes tokens 0, 2, and 3, so the three rows of its output are added to rows 0, 2, and 3 of `y` with weights 0.8808, 0.2689, and 0.1192. Combining all experts gives $y_0 = 0.8808\, f_0(x_0) + 0.1192\, f_1(x_0)$, and similarly for the other tokens.

## Constraints

- $1 \le N \le 256$, $2 \le D \le 32$, $1 \le E \le 8$, and $1 \le k \le E$.
- The output is compared against the dense evaluation of the same layer with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. An expert's output for a given row is independent of the other rows in the same call, up to floating-point rounding.
- **Cost (checked in every test):** `expert_rows` must equal $k \cdot N$ exactly, and `expert_calls` must be at most the number of active experts (which, for a correct output, equals that number).
- In every test, the $k$-th and $(k+1)$-th largest router logits for a token are either exactly equal (small-integer inputs where float32 is exact) or separated by more than $10^{-3}$, so floating-point rounding cannot alter the routing.
- A Python loop over experts is expected. No loop over tokens is needed.

## Why this matters

Sparsity is the entire reason mixture-of-experts models exist. Mixtral 8x7B activates 2 of its 8 expert MLPs per token, and Qwen3-30B-A3B activates 8 of 128, so a dense evaluation wastes 4x and 16x the intended MLP compute.

Inference engines realize those savings through this gather-run-scatter pattern, and real bugs mirror the failure modes tested here. Calling an expert one token at a time gives the right numbers but ruins GPU throughput, because hardware needs large matrix products rather than thousands of single-row calls. Calling idle experts with empty batches pays kernel launch overhead on every unused expert, which adds up quickly with 128 experts and small batches. Scattering into a contiguous slice assumes an expert's tokens are adjacent, which passes toy tests on uniform inputs and silently corrupts hidden states across unrelated tokens in real batches. Grouping tokens by expert is also the foundation of expert parallelism, where each group is dispatched to the device hosting that expert.
