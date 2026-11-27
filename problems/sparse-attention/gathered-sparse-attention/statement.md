Implement the attention half of a sparse attention layer, where each query attends only to the tokens selected for it by an indexer. Because masked full attention produces the same numerical output, the keys and values sit behind a counting wrapper that enforces reading only the selected tokens.

![This problem is the gather and the attention, which read only the rows of the cache that the indices name.](assets/map.svg)

## Specification

```python
def sparse_attend(q, store, idx): ...  # -> float32 (T, H, d)
```

There are $T$ queries with $H$ heads of width $d$, and a store holding $S$ tokens with keys $K$ and values $V$ (one key and one value per head). Row $t$ of `idx` lists up to $k$ token indices for query $t$. Every entry of `idx` is either `-1` (an **unused slot**) or a token index in $[0, S)$. Unused slots may appear anywhere in a row, and a row may contain no unused slots or only unused slots. A single row never repeats a token index, though different rows may select the same token. Slot order within a row does not affect the result.

Let $J_t = \{\, j : \texttt{idx}[t, j] \ge 0 \,\}$ be the used slots of query $t$, and let $s_j = \texttt{idx}[t, j]$ be the token in slot $j$. For every query $t$ and head $h$, compute scaled dot-product attention over the used slots:

$$
a_{t,h,j} = \frac{\exp\!\big(q_{t,h} \cdot K_{s_j,h} / \sqrt{d}\big)}
{\sum_{l \in J_t} \exp\!\big(q_{t,h} \cdot K_{s_l,h} / \sqrt{d}\big)}
\quad (j \in J_t), \qquad
\text{out}_{t,h} = \sum_{j \in J_t} a_{t,h,j}\, V_{s_j,h} .
$$

An unused slot has weight zero. A query whose slots are all unused returns all zeros. Do not modify `q` or `idx` in place.

**The store** is a [`CountingKV`](../../../python/src/learnai/kit/sparse.py) instance and provides the only access to keys $K$ and values $V$:

- `store.gather(rows, valid=None)` takes an integer tensor `rows` of any shape and returns `(k_rows, v_rows)`, each of shape `(*rows.shape, H, d)`, holding the keys and values of the listed tokens. **Every entry in `rows` counts as one row read**, even when it repeats a token.
- Indices follow Python rules, so `-1` reads and counts the **last** token. An index outside $[-S, S)$ raises `ValueError`.
- When `valid` is provided as a boolean tensor with the same shape as `rows`, every position where `valid` is `False` is not read, is not counted (regardless of its index), and comes back as zeros.
- `store.n` equals $S$, the number of tokens in the store.

**Cost.** `store.counters.rows_read` may be at most the total number of used slots, $\sum_t |J_t|$. Reading the entire store costs $S$ rows.

![In the example below, `idx` names tokens 0, 2, and 1, so the gather reads three rows and skips the `-1` slot and token 3.](assets/sparse-gather.svg)

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, result | `(T, H, d)` | float32 |
| keys and values in `store` | `(S, H, d)` | float32 |
| `idx` | `(T, k)` | integer |
| `k_rows`, `v_rows` from `store.gather(idx)` | `(T, k, H, d)` | float32 |

## Example

Consider two queries, one head of width 2, four tokens in the store, and two slots per query.

```python
q = [[[2.0, 0.0]], [[0.0, 2.0]]]
keys = [[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]], [[1.0, 1.0]]]
values = [[[1.0, 0.0]], [[0.0, 1.0]], [[3.0, 3.0]], [[-1.0, 2.0]]]
store = CountingKV(fw, keys, values)
idx = [[0, 2], [-1, 1]]

sparse_attend(q, store, idx)
# [[[1.111614, 0.167422]], [[0.0, 1.0]]]
store.counters.rows_read  # 3
```

Query 0 attends to tokens 0 and 2. Its scaled dot-product scores are $2/\sqrt{2} = 1.4142$ and $-2/\sqrt{2} = -1.4142$, which produce weights $(0.944193, 0.055807)$ and output $0.944193 \cdot (1, 0) + 0.055807 \cdot (3, 3)$.

Query 1 has one used slot, token 1, so its output is that token's value, $(0, 1)$. If `-1` were passed as a raw index, it would fetch the last token (token 3), whose key scores as high as token 1's key and would pull the output to $(-0.5, 1.5)$.

Three slots are used across both queries, so at most 3 rows may be read. Token 3 is never selected and never read.

## Constraints

- $1 \le T \le 16$, $1 \le k \le 16$, $1 \le H \le 4$, $1 \le d \le 64$, and $1 \le S \le 256$.
- The output is compared against masked full attention computed in float64 (relative tolerance `1e-4`, absolute tolerance `1e-5`).
- Every test checks that `rows_read` is at most the number of used slots. In every case except one, the store holds more tokens than there are used slots, so reading every key exceeds the limit. (In the one remaining case, every visible token is selected, so both costs coincide.)
- `gather` and `n` are the only allowed members of `store`. Accessing `counters`, `reference`, `with_framework`, or any private attribute, or setting attributes on `store`, is rejected by the harness.
- When every query selects all the tokens it is allowed to see, the result equals causal attention.

## Why this matters

DeepSeek Sparse Attention keeps a fixed number of tokens per query (2048 in GLM-5.3) regardless of context length. The compute and memory savings exist only in the kernel that gathers those rows. Reference implementations often score every key and apply a mask, which produces the right numbers while costing as much as dense attention.

Padded index tensors are also a common failure point during kernel ports. Passing a `-1` sentinel as an index silently reads the last row of the cache, and leaving a zero-filled padded slot unmasked gives it non-zero attention weight.
