Compute scaled dot-product attention for new tokens attending over a key/value buffer using MLX's fused kernel, `mx.fast.scaled_dot_product_attention`. Your main task is choosing the right mask (`None`, `"causal"`, or an explicit array) for the buffer's layout, since each form fails silently when paired with the wrong layout.

## Specification

```python
def attend(q, k, v, n_cached, causal): ...
```

A batch of `B` sequences processes `T` new tokens that attend over a preallocated buffer of `S` key/value slots, which already holds `n_cached` previously cached tokens and the `T` new tokens (`S >= n_cached + T`). Each query has `N_q` heads of dimension `D`, and the key/value buffer has `N_kv` heads of dimension `D`.

- `q`: queries for the `T` new tokens.
- `k`, `v`: key and value buffers of `S` slots filled from the front.
  - Slots $0, \dots, \texttt{n\_cached} - 1$ hold previously cached tokens.
  - Slots $\texttt{n\_cached}, \dots, \texttt{n\_cached} + T - 1$ hold the `T` new tokens (already written into the buffer).
  - Any remaining slots are **unused capacity**. They contain arbitrary finite numbers and must not affect the output.
- `n_cached`: a non-negative Python `int` giving the number of previously cached tokens.
- `causal`: a Python `bool`. When `True`, new token $i$ ($0 \le i < T$) attends to slots $j \le \texttt{n\_cached} + i$ (all cached tokens, earlier new tokens, and itself). When `False`, every new token attends to all $\texttt{n\_cached} + T$ valid slots.
- Returns the attention output with the same shape and dtype as `q`.

For a single query head and its paired key/value head, let $A_i$ be the set of valid slot indices that new token $i$ may attend to. The attention output for token $i$ is the weighted sum of values over $A_i$:

$$
\text{out}_i = \sum_{j \in A_i} a_{ij}\, v_j ,
\qquad
a_{ij} = \frac{\exp\!\left(q_i \cdot k_j / \sqrt{D}\right)}{\sum_{l \in A_i} \exp\!\left(q_i \cdot k_l / \sqrt{D}\right)} .
$$

**Grouped-query heads.** $N_{kv}$ always divides $N_q$. When $N_{kv} < N_q$, consecutive query heads share a key/value head so that query head $h$ pairs with key/value head $\lfloor h / (N_q / N_{kv}) \rfloor$.

**Fused kernel requirement.** Every call to `attend` must invoke `mx.fast.scaled_dot_product_attention` **exactly once** for the entire batch and all heads. The judge counts kernel calls. In MLX 0.32.4, the kernel signature is:

```python
mx.fast.scaled_dot_product_attention(q, k, v, *, scale, mask=None)
```

- `scale` is required and has no default. Pass $1 / \sqrt{D}$ explicitly.
- `k` and `v` may have fewer heads than `q`. The kernel pairs them using the grouped-query rule above.
- `mask=None` lets every query attend to all `S` slots.
- `mask="causal"` uses lower-right alignment where the last query aligns with the last key. The last query sees all `S` slots, the query before it sees all but the last slot, and so on.
- `mask` as a boolean array uses `True` for "may attend" and must broadcast to `(B, N_q, T, S)`.
- `mask` as a float array is added to the raw scores before softmax (`0` to attend, `-inf` to hide). Its dtype must not be wider than the dtype of `q`, `k`, and `v`.

![Two new tokens after three cached ones. The string mask is right in a full buffer, but with unused capacity it lets them read slots that hold no token.](assets/mask-alignment.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `q` | `(B, N_q, T, D)` | `float32` or `float16` |
| `k`, `v` | `(B, N_kv, S, D)`, $S \ge$ `n_cached` $+\,T$ | same dtype as `q` |
| result | `(B, N_q, T, D)` | same dtype as `q` |

## Example

Consider one sequence, one head, $D = 2$, one cached token, and two new tokens in an exact-fit buffer of three slots:

```python
q = [[[[1.0, 0.0], [0.0, 2.0]]]]  # (1, 1, 2, 2)
k = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]  # (1, 1, 3, 2)
v = [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]]]

attend(q, k, v, n_cached=1, causal=True)
# [[[[0.6698, 0.3302],
#    [1.0000, 1.3374]]]]
```

- New token 0 attends to slots 0 and 1. Its scaled scores are $[1, 0] / \sqrt{2} = [0.7071, 0]$, giving softmax weights $[0.6698, 0.3302]$ and output $0.6698\,[1, 0] + 0.3302\,[0, 1] = [0.6698, 0.3302]$.
- New token 1 attends to all three slots. Its scaled scores are $[0, 2, 2] / \sqrt{2} = [0, 1.4142, 1.4142]$, giving softmax weights $[0.1084, 0.4458, 0.4458]$ and output $[0.1084 + 0.8916,\; 0.4458 + 0.8916] = [1.0000, 1.3374]$.

Now place the same three tokens into a five-slot buffer with arbitrary values in the last two slots:

```python
k = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [9.0, 9.0], [-7.0, 3.0]]]]
v = [[[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [50.0, -50.0], [80.0, 80.0]]]]

attend(q, k, v, n_cached=1, causal=True)  # the same result
```

In this padded buffer, `mask="causal"` would align with slot 4 instead of slot 2, letting token 0 read slots 0 to 3 and token 1 read slots 0 to 4.

## Constraints

- $1 \le B \le 4$, $1 \le N_{kv} \le N_q \le 8$, $1 \le T \le 24$, $T \le S \le 48$, $D \in \{2, 4, 8, 64\}$.
- `float32` results are checked against a `float64` reference with relative tolerance `1e-4` and absolute tolerance `1e-5`. `float16` results are checked with relative tolerance `1e-2` and absolute tolerance `2e-2`. The output must match the dtype of `q`.
- `mx.fast.scaled_dot_product_attention` must be called exactly once per call to `attend`. Computing attention manually with `matmul` and `softmax` fails even if the numbers match.
- No Python loops over batch items, heads, or positions are needed.

## Why this matters

Attention dominates memory bandwidth during decoding and prefill. A hand-written implementation materializes a `(T, S)` score matrix for every head of every layer. For a 2,048-token prefill with 8 heads in `float32`, those scores alone take 128 MiB per layer (136 MiB peak on Apple Silicon), whereas the fused kernel never materializes them and peaks at 4 MiB.

Production libraries like `mlx-lm` rely on this fused kernel, and the mask is the one argument that changes with cache layout. A growing cache is always full, so `"causal"` (or `None` for a single decode token) works as expected. A preallocated fixed-size buffer, needed when compiling a decode step with static shapes, leaves unused slots at the end. Passing `"causal"` to a partially filled buffer silently lets queries attend to zeros or stale data from an earlier request, producing plausible-looking but corrupted output with no error.
