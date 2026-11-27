Run one sparse-attention layer on a chunk of new tokens that continue a cached sequence. The layer maintains two caches—one for the indexer's keys and one for attention's keys and values—and updates both before scoring any new query.

![This problem is the two caches, the indexer's keys and the attention's keys and values, which grow with each call.](assets/map.svg)

## Specification

```python
def dsa_step(dsa, layer, x_new, cache, k): ...  # -> (out, cache)
```

`dsa` is a [`TinyDSA`](../../../python/src/learnai/kit/sparse.py) stack of sparse-attention layers exposed through four row-wise methods. For an input tensor `x` of shape `(n, d_model)`, where `n` is the number of tokens and `d_model` is the model width:

```python
k_idx = dsa.index_key(layer, x)       # (n, index_dim); counted
q_idx, w = dsa.index_query(layer, x)  # (n, index_heads, index_dim), (n, index_heads)
q, k, v = dsa.qkv(layer, x)           # (n, n_heads, head_dim) each
y = dsa.out(layer, attended)          # (n, n_heads, head_dim) -> (n, d_model)
```

**What the layer computes.** Given a sequence $x_0, \dots, x_{n-1}$, let $H = \text{dsa.index\_heads}$ and $D = \text{dsa.index\_dim}$, and let $k^I_s$, $q^I_{t,h}$, and $w_{t,h}$ denote the indexer keys (`k_idx`), indexer queries (`q_idx`), and head weights (`w`). For each token position $t$:

1. **Scores.** Compute the index score $I_{t,s}$ against every token $s \le t$ using the `indexer-topk-select` formula:
   $$I_{t,s} = \sum_{h=0}^{H-1} \frac{w_{t,h}}{\sqrt{H}} \operatorname{relu}\!\left(\frac{q^I_{t,h} \cdot k^I_s}{\sqrt{D}}\right)$$
2. **Selection.** Choose the $\min(k, t + 1)$ tokens $s \le t$ with the highest scores. Break ties in favor of the lower token index. Token $t$ is a candidate for itself.
3. **Attention.** Run scaled dot-product attention per head over the selected tokens only, using scale `dsa.head_dim ** -0.5` as in `gathered-sparse-attention`. Then project the result with `dsa.out`.

**What a call does.** The incoming `cache` holds the $S$ tokens already processed, and `x_new` holds the next $T$ tokens. New row $i$ corresponds to token position $S + i$. Return rows $S, \dots, S + T - 1$ of the layer output above along with an updated cache holding all $S + T$ tokens. Any chunking of the same sequence must produce the same outputs.

**The cache** is a pair `(index_keys, store)`:

- `index_keys` stores one indexer key per cached token, where row `s` belongs to token `s`.
- `store` is a `CountingKV` holding the attention keys and values for the $S$ cached tokens. Its length is `store.n == S`. Calling `store.append(k, v)` returns a longer store without modifying the original. Calling `store.gather(rows, valid=None)` reads and counts rows as in `gathered-sparse-attention`, where `-1` reads the last row and slots with `valid == False` are not read.

Before the first call, `index_keys` has shape `(0, index_dim)` and `store` is empty. The returned cache must contain `index_keys` of shape `(S + T, index_dim)` and the store returned by `store.append`.

![The two calls of the example below, with $k = 2$: both caches hold a new token before its query is scored.](assets/cache-steps.svg)

**Cost rules.**

- `dsa.counters.index_key_rows` must equal the total number of tokens processed. Compute each token's indexer key once, in the call that introduces that token.
- Each call may read at most $\sum_{i=0}^{T-1} \min(k,\, S + i + 1)$ rows from `store`, which allows one read per selected token.
- Each call must read at least one row from `store` for every distinct token from **earlier calls** that its queries select. Those keys and values must come from `store`, not from a side copy.

**Arguments and mutability.**

- `layer`: Python `int` in `[0, dsa.n_layers)`.
- `x_new`: input chunk of $T \ge 1$ new tokens.
- `k`: Python `int` with $k \ge 1$. It may exceed $S + T$.
- Return `(out, cache)`.
- Do not modify `x_new` or the input `cache` in place, so the caller can continue from the same cache again.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x_new`, `out` | `(T, d_model)` | float32 |
| `index_keys` (in, out) | `(S, index_dim)`, `(S + T, index_dim)` | float32 |
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

In the first call, token 0 can select only itself and token 1 selects both tokens. That call computes 2 indexer keys and reads 3 rows from `store`.

In the second call, token 2 scores $(0.2708, -0.3701, 0.4762)$ against tokens 0, 1, and 2. It selects tokens 2 and 0, reading 2 rows. Token 2's own indexer key must already be in `index_keys` before scoring, or else it would only see tokens 0 and 1 and select both of them. For comparison, dense attention over all three tokens would return $(-0.0422, 0.0261, 0.0321, 0.0050)$.

## Constraints

- Sequences have at most 16 tokens, passed in chunks of $1 \le T \le 16$, with $1 \le k \le 64$, at most 4 indexer heads, and `d_model` $\le 32$.
- Outputs, the returned `index_keys`, and the contents of the returned `store` are checked against the kit's full causal pass with relative tolerance `1e-4` and absolute tolerance `1e-5`.
- In every test sequence, any two index scores in the same row that decide a selection are either exactly equal or more than `1e-4` apart.
- The cost limits are checked on every call. Reading the entire cache ($S + T$ rows) violates the upper limit whenever $S + T > k\,T$. The keys and values of the call's own tokens are already available before being appended and do not need to be read back.
- Use only the interface methods listed above. The harness rejects `dsa.indexer`, `dsa.attend`, all `*_ref` methods, `store.reference()`, `with_framework`, accessing counters or private attributes on either object, and setting attributes on either object.

## Why this matters

Models such as GLM-5.3 and DeepSeek-V3.2 store indexer keys alongside attention keys and values in each layer's cache and extend both on every step. The indexer cache is easy to get wrong. If it lags by one token, a query can never select itself. If a prefill chunk is scored without a causal mask, earlier tokens in the chunk select future tokens. In both cases the model still generates fluent text from the wrong context, so the bug only shows up when you compare cached decoding against an uncached pass.
