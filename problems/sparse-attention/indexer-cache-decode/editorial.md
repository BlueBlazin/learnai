## The idea

A dense attention layer with a KV cache has one invariant: after a call, the cache
holds the keys and values of every token seen so far. A sparse layer has that
invariant twice. The indexer scores a query against one indexer key per earlier
token, and recomputing those keys would need the earlier inputs, which are gone. So
the layer caches them, and now two stores describe the same token sequence:

| cache | one row per token | used for |
|---|---|---|
| `index_keys` | `(index_dim,)` | scoring every earlier token, cheaply |
| `store` | keys and values `(n_heads, head_dim)` | attention over the selected rows |

Row `s` of both is token `s`. Everything in this problem follows from keeping that
true at the moment each is used:

1. **Append, then score.** A token is a candidate for its own selection, and a new
   token is a candidate for the new tokens after it. So the new indexer keys go into
   `index_keys` before any new query is scored, and the new keys and values go into
   the store before anything is gathered, because the indices the selection produces
   point into the grown store.
2. **The mask is about positions, not about the call.** New row `i` is token `S + i`
   and may select columns `0 .. S + i`. Of the two shortcuts, "everything is visible"
   is right only for `T = 1`, and "column `s <= i`" is right only for `S = 0`. A test
   that prefills in one call from an empty cache hides the second, and token-by-token
   decoding hides the first.
3. **One selection per query.** The rows of a chunk select different tokens.

With those three in place, a call computes exactly the rows `S .. S + T - 1` of the
full causal pass: token `t`'s scores, mask, selection, and attention depend only on
tokens `0 .. t`, and all of them are in the caches when row `t` is processed. That is
the whole argument for "any chunking gives the same outputs".

## Reference solution (PyTorch)

```python
def dsa_step(dsa, layer, x_new, cache, k):
    index_keys, store = cache
    seen, new = store.n, x_new.shape[0]

    index_keys = torch.cat((index_keys, dsa.index_key(layer, x_new)))
    q, k_new, v_new = dsa.qkv(layer, x_new)
    store = store.append(k_new, v_new)

    q_idx, w = dsa.index_query(layer, x_new)
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, index_keys) * dsa.index_dim**-0.5)
    scores = torch.einsum("th,ths->ts", w * dsa.index_heads**-0.5, per_head)

    positions = seen + torch.arange(new)
    visible = torch.arange(seen + new)[None, :] <= positions[:, None]
    masked = torch.where(visible, scores, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)[:, :k]
    idx = torch.where(torch.gather(visible, 1, order), order, -1)

    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)
    attn = torch.einsum("thd,tjhd->thj", q, k_rows) * dsa.head_dim**-0.5
    attn = torch.softmax(torch.where(valid[:, None, :], attn, float("-inf")), dim=-1)
    out = dsa.out(layer, torch.einsum("thj,tjhd->thd", attn, v_rows))
    return out, (index_keys, store)
```

Every row has at least one visible token, itself, so no softmax is empty. `[:, :k]`
handles `k > S + T`; picks beyond a row's visible tokens sit at `-inf` and become `-1`.
`torch.cat` and `store.append` build new objects, so the cache passed in stays valid.

## Common mistakes

- **Scoring before the indexer cache has grown.** The queries see the old keys only.
  A single new token can then never select itself; in the worked example token 2
  would select tokens 0 and 1 instead of 2 and 0. The first token of a sequence has
  nothing to select at all.
- **Forcing a token into its own selection.** A token is a candidate for itself, not
  a member by right: when its own score is not among the `k` best, it is left out.
- **No causal mask inside a call.** Every new query may select every new token.
  Invisible in token-by-token decoding, wrong for every chunked prefill.
- **The mask without the offset** (`s <= i`). Row `i` then sees `i + 1` of the cached
  tokens and nothing else. Invisible when the cache is empty.
- **One selection for the chunk,** computed for its last query.
- **The index cache not returned grown.** This call is right and the next one is
  not: the scores have fewer columns than the store has rows.
- **Indexer keys computed twice,** once to score and once to store, or recomputed in
  a loop over the chunk. Correct output, twice the indexer work. In a real model the
  key is a projection and a LayerNorm per token per layer.
- **Reading the whole cache** and masking. Correct output at the cost of dense
  attention.
- **Padding read as the last row.** `store.gather(idx)` with `-1` in it reads the
  newest token for every missing pick. Masked afterward the output is right, and the
  early queries of a sequence read `k` rows where one or two exist.
- **Ties the other way.** Exact zeros are common after the ReLU; see
  `indexer-topk-select`.
- **A copy of the cache kept elsewhere** (a module-level dictionary, an attribute on
  the store) and attended densely. No row is read, so the call fails the lower
  limit: tokens of earlier calls have to come from the store.
- **Editing the counters,** or reading through a `with_framework` twin. Rejected by
  the harness.
- **The kit's `indexer` and `attend`.** They compute the uncached layer and need the
  whole sequence; the harness rejects them.

## Complexity

For a call of $T$ tokens after $S$: the indexer costs $O(T\,(S + T)\,H\,D)$ and the
attention $O(T\,k)$ rows of width `n_heads * head_dim`. Per decoded token that is
linear in the context for the indexer (with the small constant $H \cdot D$) and
constant for the attention. Memory grows by `index_dim` numbers per token per layer on
top of the KV cache.

The row limit cannot see one thing. The keys and values of the call's own tokens are
in hand before they enter the store, so a single prefill call could attend densely
within itself without reading. The tests therefore check the limit per call, and the
single-token calls after a long prefix are where reading everything fails.

## In the wild

Checked on 2026-11-21 in `transformers` 5.23.0.

- `GlmMoeDsaIndexer.forward` (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) computes
  the new tokens' indexer keys `k_norm(wk(hidden_states))`, calls
  `past_key_values.update_indexer(k, layer_idx)`, and scores the new queries against
  what that returns. `DynamicIndexedLayer.update_indexer` (`cache_utils.py`)
  concatenates the new keys onto `indexer_keys` and returns the full tensor: append,
  then score. The mask it applies is the model's causal mask for the new rows against
  all positions, so causality inside a chunk comes from the same mask as in dense
  attention. `GlmMoeDsaAttention.forward` updates the main cache
  (`past_key_values.update`) before it calls the indexer.
- The kit's `TinyDSA` was compared with that module by the kit's tests: index scores
  to `6e-7`, and the selections of a run with the shipped `DynamicCache` over chunks
  of 6, 1, 1, 8, and 8 tokens against the kit's full pass.

What this problem leaves out of the shipped layer: the batch axis; rotary positions
on the first `qk_rope_head_dim` channels of the indexer's queries and keys (the cached
indexer keys are stored already rotated); the low-rank query residual the shipped
indexer projects its queries from (`TinyDSA` projects them from `x`); the latent form
of the main cache (GLM caches one compressed vector per token and expands it to keys
and values, where the store here holds per-head rows); and the form of the
selection (the shipped eager path scatters the indices into a mask over all keys
and reads every key; see `gathered-sparse-attention`). `TinyDSA`'s main attention and
output projection are stand-ins that were not compared with a shipped model. The tie
rule is this track's own: `torch.topk` does not order equal scores.
