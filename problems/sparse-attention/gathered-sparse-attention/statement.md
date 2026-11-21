Implement the attention half of a sparse attention layer: each query attends to the
tokens an indexer selected for it and to nothing else. Masked full attention returns
the same numbers, so the keys and values sit behind a counter and the unselected ones
must never be read.

![This problem is the gather and the attention, which read only the rows of the cache that the indices name.](assets/map.svg)

## Specification

```python
def sparse_attend(q, store, idx): ...  # -> float32 (T, H, d)
```

There are $T$ queries with $H$ heads of width $d$, and a store of $S$ tokens, each
with one key and one value per head. Row $t$ of `idx` lists the tokens query $t$
attends to, in $k$ slots; an entry of `-1` marks an **unused slot**. Let
$J_t = \{\, j : \texttt{idx}[t, j] \ge 0 \,\}$ be the used slots of query $t$ and
$s_j = \texttt{idx}[t, j]$ the token in slot $j$. For every query $t$ and head $h$:

$$
a_{t,h,j} = \frac{\exp\!\big(q_{t,h} \cdot K_{s_j,h} / \sqrt{d}\big)}
{\sum_{l \in J_t} \exp\!\big(q_{t,h} \cdot K_{s_l,h} / \sqrt{d}\big)}
\quad (j \in J_t), \qquad
\text{out}_{t,h} = \sum_{j \in J_t} a_{t,h,j}\, V_{s_j,h} .
$$

An unused slot has weight zero. A query whose slots are all unused returns zeros.

**The store** is a [`CountingKV`](../../../python/src/learnai/kit/sparse.py). It is the
only access to the keys $K$ and values $V$:

- `store.gather(rows, valid=None)`: `rows` is an integer tensor of any shape. Returns
  `(k_rows, v_rows)`, each of shape `(*rows.shape, H, d)`: the keys and the values of
  the listed tokens. **Every entry of `rows` counts as one row read**, also when it
  repeats a token.
- Indices follow Python: `-1` is the **last** token, read and counted like any other.
  An index outside $[-S, S)$ raises `ValueError`.
- `valid`, if given, is a boolean tensor of the shape of `rows`. Where it is False the
  slot is not read and not counted, whatever its index, and comes back as zeros.
- `store.n` is $S$.

**Cost.** `store.counters.rows_read` may be at most the number of used slots,
$\sum_t |J_t|$. Reading the whole store costs $S$ rows.

![In the example below, `idx` names tokens 0, 2, and 1, so the gather reads three rows and skips the `-1` slot and token 3.](assets/sparse-gather.svg)

Arguments:

- `q`: float32, shape `(T, H, d)`.
- `store`: as above, holding float32 keys and values of shape `(S, H, d)`.
- `idx`: integer tensor, shape `(T, k)`. Every entry is `-1` or a token index in
  $[0, S)$. Unused slots may be anywhere in a row, and a row may have none or only
  unused slots. A row never lists a token twice; two rows may list the same token.
  The order of the slots in a row does not affect the result.
- Returns float32, shape `(T, H, d)`.
- `q` and `idx` are not modified.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q`, result | `(T, H, d)` | float32 |
| keys and values in `store` | `(S, H, d)` | float32 |
| `idx` | `(T, k)` | integer |
| `k_rows`, `v_rows` from `store.gather(idx)` | `(T, k, H, d)` | float32 |

## Example

Two queries, one head of width 2, four tokens in the store, two slots:

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

Query 0 attends to tokens 0 and 2. Its scores are $2/\sqrt{2} = 1.4142$ and
$-2/\sqrt{2} = -1.4142$, so the weights are $(0.944193, 0.055807)$ and the output is
$0.944193 \cdot (1, 0) + 0.055807 \cdot (3, 3)$.

Query 1 has one used slot, token 1, so its output is that token's value, $(0, 1)$.
Read as an index, its `-1` would fetch the last token, whose key scores as high as
token 1's, and the output would be $(-0.5, 1.5)$. Three slots are used, so at most 3
rows may be read; token 3 is never selected and never read.

## Constraints

- $1 \le T \le 16$, $1 \le k \le 16$, $1 \le H \le 4$, $1 \le d \le 64$,
  $1 \le S \le 256$.
- The output is compared with masked full attention computed in float64: relative
  tolerance `1e-4`, absolute tolerance `1e-5`.
- **Cost, checked in every case:** `rows_read` at most the number of used slots. In
  all cases but one the store holds more tokens than the call has used slots, so
  reading every key exceeds the limit. (When every visible token is selected, as in
  one case, the two costs coincide.)
- `gather` and `n` are the whole interface of the store. Its counters, its
  `reference` and `with_framework` methods, and its private attributes are off
  limits, and no attribute may be set on it; the harness rejects each of these.
- When every query selects all the tokens it may see, the result equals causal
  attention.

## Why this matters

DeepSeek Sparse Attention keeps a fixed number of tokens per query (2048 in GLM-5.3)
however long the context is, and the saving exists only in the kernel
that gathers those rows: the reference path of the same models scores every key and
masks, which is correct and costs as much as dense attention. The padded index tensor
is where ports break. A sentinel used as an index reads a real row, and a padded slot
that is read as zeros and not masked still receives weight.
