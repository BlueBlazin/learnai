Run one sparse-attention layer on a chunk of new tokens that continue a cached
sequence. The layer keeps two caches, the indexer's keys and the attention's keys and
values, and both must hold the new tokens before the first new query is scored.

![This problem is the two caches, the indexer's keys and the attention's keys and values, which grow with each call.](assets/map.svg)

## Specification

```python
def dsa_step(dsa, layer, x_new, cache, k): ...  # -> (out, cache)
```

`dsa` is a [`TinyDSA`](../../../python/src/learnai/kit/sparse.py), a stack of
sparse-attention layers handed out in pieces. For rows `x` of shape `(n, d_model)`,
each piece works row by row:

```python
k_idx = dsa.index_key(layer, x)  # (n, index_dim): one indexer key per token; counted
q_idx, w = dsa.index_query(layer, x)  # (n, index_heads, index_dim), (n, index_heads)
q, k, v = dsa.qkv(layer, x)  # (n, n_heads, head_dim) each
y = dsa.out(layer, attended)  # (n, n_heads, head_dim) -> (n, d_model)
```

**What the layer computes.** For a sequence $x_0, \dots, x_{n-1}$, token $t$:

1. **Scores.** $I_{t,s}$ for every $s \le t$, by the formula of `indexer-topk-select`
   with $H$ = `dsa.index_heads` and $D$ = `dsa.index_dim`:
   $I_{t,s} = \sum_h (w_{t,h} / \sqrt{H}) \operatorname{relu}(q^I_{t,h} \cdot k^I_s / \sqrt{D})$.
2. **Selection.** The $\min(k, t + 1)$ tokens $s \le t$ with the highest scores; among
   equal scores the lower index first. Token $t$ is a candidate for itself.
3. **Attention** over the selected tokens only, per head, with scale
   `dsa.head_dim ** -0.5`, as in `gathered-sparse-attention`; then `dsa.out`.

**What a call does.** `cache` describes the $S$ tokens processed so far, and `x_new`
holds the next $T$: new row $i$ is token $S + i$. The call returns rows
$S, \dots, S + T - 1$ of the result above and a cache for $S + T$ tokens. However a
sequence is cut into calls, the outputs are the same.

**The cache** is a pair `(index_keys, store)`:

- `index_keys`: float32, shape `(S, index_dim)`; row `s` is the indexer key of token `s`.
- `store`: a `CountingKV` holding the keys and values of the $S$ tokens, `(S, n_heads,
  head_dim)` each. `store.n` is $S$; `store.append(k, v)` returns a longer store and
  leaves the given one unchanged; `store.gather(rows, valid=None)` reads rows and
  counts them, as in `gathered-sparse-attention` (`-1` reads the last row; a slot with
  `valid` False is not read).

Before the first call, `index_keys` has shape `(0, index_dim)` and the store is empty.
The returned cache holds `index_keys` of shape `(S + T, index_dim)` and the store
obtained from the given one by `append`.

![The two calls of the example below, with $k = 2$: both caches hold a new token before its query is scored.](assets/cache-steps.svg)

**Cost.**

- `dsa.counters.index_key_rows` must equal the number of tokens processed: each
  token's indexer key is computed once, in the call that brings the token.
- A call may read at most $\sum_{i < T} \min(k,\, S + i + 1)$ rows from the store: one
  per selected token.
- A call must read at least one row for every distinct token of **earlier calls**
  that its queries select. Those keys and values are to be taken from the store,
  not from a copy kept elsewhere.

Arguments:

- `layer`: a Python int in `[0, dsa.n_layers)`.
- `x_new`: float32, shape `(T, d_model)`, $T \ge 1$.
- `k`: a Python int, $k \ge 1$; it may exceed $S + T$.
- Returns `(out, cache)` with `out` float32 of shape `(T, d_model)`.
- `x_new` and the cache passed in are not modified: the same cache can be continued
  again.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x_new`, `out` | `(T, d_model)` | float32 |
| `index_keys` in, out | `(S, index_dim)`, `(S + T, index_dim)` | float32 |
| keys and values in `store` | `(S, n_heads, head_dim)` | float32 |

## Example

```python
dsa = TinyDSA(fw, seed=9, n_layers=1, d_model=4, n_heads=1, head_dim=2, index_heads=2, index_dim=4)
x = [[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 1.0, -1.0], [-1.5, 1.0, 0.0, 1.0]]
cache = (zeros((0, 4)), CountingKV.empty(fw))

out, cache = dsa_step(dsa, 0, x[:2], cache, 2)
# [[-0.167168, 0.104263, 0.128440, 0.019116],
#  [-0.447671, 0.036697, 0.081901, 0.244704]]
out, cache = dsa_step(dsa, 0, x[2:], cache, 2)
# [[0.178081, 0.025419, 0.010662, -0.129273]]
```

In the first call, token 0 can select only itself, and token 1 selects both tokens:
3 rows read, 2 indexer keys computed. In the second call the scores of token 2 are
$(0.2708, -0.3701, 0.4762)$ for tokens 0, 1, and 2: it selects itself and token 0, and
reads 2 rows. Its own key had to be in the cache before it was scored; scored against
the two cached keys only, it would select tokens 0 and 1. Dense attention over all
three tokens would return $(-0.0422, 0.0261, 0.0321, 0.0050)$.

## Constraints

- A sequence has at most 16 tokens, in calls of $1 \le T \le 16$; $1 \le k \le 64$;
  `dsa` has at most 4 indexer heads and `d_model` $\le 32$.
- Outputs, the returned `index_keys`, and the contents of the returned store are
  compared with the kit's full causal pass: relative tolerance `1e-4`, absolute
  tolerance `1e-5`.
- In every test sequence, two index scores of one row that decide its selection are
  either exactly equal or more than `1e-4` apart.
- The cost limits above are checked in every call. A call that reads the whole cache
  ($S + T$ rows) exceeds its upper limit whenever $S + T > k\,T$. The keys and values
  of the call's own tokens are in hand before they are stored and need not be read
  back.
- The methods named above are the whole interface. Not allowed, and rejected by the
  harness: `dsa.indexer`, `dsa.attend`, the `*_ref` methods, `store.reference()`,
  `with_framework`, the counters and private attributes of either object, and
  setting an attribute on either.

## Why this matters

GLM-5.3 and DeepSeek-V3.2 keep the indexer's keys in the same per-layer cache as the
attention's keys and values and extend both on every call. The indexer cache is the
one people forget: when it lags by one token, a query cannot select itself, and when a
prefill chunk is scored without a causal mask of its own, its tokens select the
future. Either way the model produces fluent text from the wrong context, and only a
comparison with the uncached pass shows it.
