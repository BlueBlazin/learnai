Run a stack of sparse-attention layers in which only some layers own an indexer. The
others attend with the token selection of the nearest indexed layer below them, so
most of the indexer work of the stack is skipped.

## Specification

```python
def forward_shared(dsa, h, types, k): ...  # -> float32 (T, d_model)
```

`dsa` is a [`TinyDSA`](../../../python/src/learnai/kit/sparse.py) with `dsa.n_layers`
sparse-attention layers. Two of its methods are all this problem needs:

```python
idx = dsa.indexer(layer, h, k)  # (T, min(k, T)) integers; one counted indexer call
out = dsa.attend(layer, h, idx)  # (T, d_model)
```

- `dsa.indexer(layer, h, k)` runs the indexer of `layer` on the whole sequence `h` and
  returns, for each token, the `k` tokens at or before it with the highest index
  scores (`-1` in slots it cannot fill). It can be called for any layer and any `h`.
- `dsa.attend(layer, h, idx)` is the attention of `layer` on `h`, where token `t`
  attends to the tokens listed in `idx[t]` only. `idx` may come from another layer's
  indexer.

`types[i]` is `"full"` or `"shared"` for layer `i`. Starting from the input `h`, for
`i = 0, 1, ..., dsa.n_layers - 1` in order:

1. **Selection.** If layer `i` is full, its selection is `dsa.indexer(i, h, k)`,
   computed from the layer's own input (the current `h`). If it is shared, its
   selection is the one used by the **nearest full layer before it**: the indices that
   layer's indexer returned, unchanged, although `h` has changed since.
2. **Attention with a residual.** `h = h + dsa.attend(i, h, idx)`.

The result is `h` after the last layer.

A shared layer has no indexer of its own and runs none. If a shared layer has no full
layer before it, raise `ValueError` yourself, before any call to `dsa.indexer` or
`dsa.attend`. (`dsa.attend` happens to raise `ValueError` when it is given no
indices; an error that comes from there does not count.)

**Cost.** `dsa.counters.indexer_calls` must equal the number of full layers.

Arguments:

- `h`: float32, shape `(T, d_model)` with `d_model = dsa.d_model`.
- `types`: a list of exactly `dsa.n_layers` strings, each `"full"` or `"shared"`. Use
  this argument; `dsa.types` is the kit's own default pattern and is not it.
- `k`: a Python int, $k \ge 1$; it may exceed $T$.
- Returns float32, shape `(T, d_model)`.
- `h` is not modified.

## Shapes

| name | shape | dtype |
|---|---|---|
| `h`, result, `dsa.attend(...)` | `(T, d_model)` | float32 |
| `dsa.indexer(...)` | `(T, min(k, T))` | integer |

## Example

Three layers, full, shared, full, on four tokens of width 4:

```python
dsa = TinyDSA(fw, seed=1, n_layers=3, d_model=4, n_heads=1, head_dim=2, index_heads=2, index_dim=4)
h = [[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 1.0, -1.0], [-1.5, 1.0, 0.0, 1.0], [0.5, -1.0, 2.0, 0.0]]

forward_shared(dsa, h, ["full", "shared", "full"], 2)
# [[ 0.929907, -0.478311, -1.247557,  0.786506],
#  [-0.392463,  2.861226,  0.909047, -1.360618],
#  [-2.234913,  1.039535, -0.015184,  1.150592],
#  [-0.029923,  0.015205,  1.958411, -0.301067]]
dsa.counters.indexer_calls  # 2
```

Layer 0 runs its indexer and selects `[[0, -1], [0, 1], [0, 2], [0, 1]]`: token 0 can
select only itself, and token 3 selects tokens 0 and 1. Layer 1 is shared and attends
with those indices. Its own indexer, on its own input, would have selected
`[[0, -1], [0, 1], [1, 0], [2, 0]]`; it is not run. Layer 2 runs its indexer on its
input and selects `[[0, -1], [1, 0], [0, 2], [0, 1]]`. With all three layers full, the
last two rows of the result would differ (row 3 would start with `0.244022`).

## Constraints

- $1 \le$ `dsa.n_layers` $\le 12$, $1 \le T \le 16$, `d_model` $\le 32$, $1 \le k \le 64$.
- The output is compared with the kit's stack oracle: relative tolerance `1e-4`,
  absolute tolerance `1e-5`.
- `indexer_calls` is compared exactly, and the tests also check, from the indices
  passed to `dsa.attend`, that each layer attended with the indices of its nearest
  full layer. For the layer types of GLM-5.3's first 11 layers, that pattern is
  compared with the one the shipped model executes.
- In every test input, two index scores of one row that decide a selection are either
  exactly equal or more than `1e-4` apart.
- `indexer`, `attend`, and the sizes of `dsa` are the whole interface. Not allowed,
  and rejected by the harness: the `*_ref` methods, `with_framework`, the counters,
  private attributes, and setting an attribute on `dsa`.

## Why this matters

The indexer is the part of sparse attention that is still quadratic in the context,
and it runs in every layer. GLM-5.3 runs it in 21 of its 78 layers: the first three,
then every fourth. The other 57 reuse the selection handed up from below. An engine
that ports this has to carry the indices across layers as state. Recomputing them
where the model was trained to reuse them gives a different model at a higher cost,
and reusing the wrong layer's indices gives a different model at the right cost, which
is harder to notice.
