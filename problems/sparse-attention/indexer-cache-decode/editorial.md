## The idea

A dense attention layer with a KV cache keeps one invariant: after each call, the cache holds the keys and values of every token seen so far. A sparse attention layer keeps that invariant for two caches. The indexer scores each query against one indexer key per earlier token, and recomputing those keys would require the earlier inputs, which are no longer available. Caching them leaves two stores describing the same sequence of $S$ cached tokens plus $T$ new tokens:

| cache | one row per token | used for |
|---|---|---|
| `index_keys` | `(index_dim,)` | scoring every earlier token cheaply |
| `store` | keys and values `(n_heads, head_dim)` | attention over the selected rows |

Row `s` in both caches corresponds to token `s`. Three rules keep that alignment intact at every step:

1. **Append, then score.** Every token is a candidate for itself, and each new token is a candidate for later tokens in the same chunk. Append the new indexer keys to `index_keys` before scoring any new query, and append the new keys and values to `store` before gathering, because the selected indices address the grown store.
2. **Mask by sequence position, not by call.** New row `i` sits at sequence position `S + i` and can see columns `0 .. S + i`. Treating every column as visible works only when `T = 1`. Masking with `s <= i` works only when `S = 0`. A single-call prefill hides the second bug, and token-by-token decoding hides the first.
3. **Select per query.** Different rows in a chunk select different sets of tokens.

With these three rules in place, each call computes rows `S .. S + T - 1` of the full causal pass. Token `t`'s scores, mask, selection, and attention depend only on tokens `0 .. t`, and all of those tokens are in both caches when row `t` runs. That is why every chunking of the sequence gives identical outputs.

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

Every row has at least one visible token (itself), so no softmax row is ever empty. Slicing `[:, :k]` handles `k > S + T`, and any pick beyond a row's visible tokens sits at `-inf` and is replaced with `-1`. Both `torch.cat` and `store.append` return new objects, so the input cache remains unchanged.

## Common mistakes

- **Scoring before growing `index_keys`.** Queries only see older keys, so a new token can never select itself. In the worked example, token 2 would select tokens 0 and 1 instead of 2 and 0, and the very first token of a sequence would have nothing to select.
- **Forcing a token into its own selection.** A token is a candidate for itself, not guaranteed to be chosen. If its score is not among the top `k`, it must be omitted.
- **Omitting the causal mask inside a chunk.** Every new query can then select future tokens in the same call. This bug is invisible during single-token decoding and fails on chunked prefill.
- **Omitting the `S` offset in the mask (`s <= i`).** Row `i` then sees only the first `i + 1` cached tokens. This bug is invisible when the cache starts empty.
- **Sharing one selection across the whole chunk.** Computing indices only for the last query produces wrong selections for earlier rows in the chunk.
- **Returning the old `index_keys`.** The current call succeeds, but the next call fails because `index_keys` has fewer rows than `store`.
- **Computing indexer keys twice.** Computing `index_key` once to score and again to cache (or recomputing inside a loop) gives the right values at double the indexer cost. In a real model, each indexer key requires a projection and a LayerNorm per token per layer.
- **Reading the entire cache and masking.** This produces the right output but pays dense attention cost and violates the row-read limit.
- **Reading `-1` padding rows as the last row.** Calling `store.gather(idx)` without `valid` reads the newest token for every unused slot. Masking afterward fixes the output, but early queries read `k` rows when only one or two exist.
- **Breaking ties in the wrong order.** Exact zeros occur often after ReLU, so unstable sorts or reverse tie-breaking pick the wrong indices (see `indexer-topk-select`).
- **Keeping a side copy of the cache.** Storing keys and values in a module-level dictionary or custom attribute and attending densely reads zero rows from `store`, which fails the lower-bound read check for tokens from earlier calls.
- **Tampering with counters or using `with_framework`.** Rejected by the harness.
- **Calling `dsa.indexer` or `dsa.attend`.** Those methods run the uncached full-sequence layer and are rejected by the harness.

## Complexity

For a call of $T$ new tokens after $S$ cached tokens, the indexer takes $O(T(S + T)HD)$ time, where $H = \text{index\_heads}$ and $D = \text{index\_dim}$. Attention reads $O(Tk)$ rows of width `n_heads * head_dim`. Per decoded token ($T = 1$), indexer cost is linear in context length with a small constant $HD$, and attention cost is constant in context length. Cache memory grows by `index_dim` float32 values per token per layer on top of the KV cache.

The per-call row limit has one blind spot: the keys and values of the current call's tokens are already in hand before they enter `store`, so a single prefill call could attend densely within itself without reading from `store`. The test suite checks the limit on every call, so single-token decode steps after a long prefix catch implementations that read the whole cache.

## In the wild

Checked on 2026-11-21 in `transformers` 5.23.0.

- `GlmMoeDsaIndexer.forward` (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) computes the new indexer keys with `k_norm(wk(hidden_states))`, calls `past_key_values.update_indexer(k, layer_idx)`, and scores new queries against the returned tensor. `DynamicIndexedLayer.update_indexer` (`cache_utils.py`) concatenates the new keys onto `indexer_keys` and returns the full tensor, following the same append-then-score order. It applies the model's causal mask for the new rows against all positions, so causality inside a chunk matches dense attention. `GlmMoeDsaAttention.forward` updates the main cache (`past_key_values.update`) before calling the indexer.
- The kit's tests compare `TinyDSA` against that module: index scores match to `6e-7`, and selections match across chunks of 6, 1, 1, 8, and 8 tokens with `DynamicCache` against the kit's full pass.

This problem omits several details of the shipped layer:

- The batch axis.
- Rotary position embeddings on the first `qk_rope_head_dim` channels of indexer queries and keys (the shipped cache stores indexer keys already rotated).
- The low-rank query residual from which the shipped indexer projects its queries (`TinyDSA` projects directly from `x`).
- The latent KV cache format (GLM caches one compressed vector per token and expands it to keys and values, whereas `store` holds per-head rows).
- The eager selection path, which scatters indices into a dense mask over all keys and reads every key (see `gathered-sparse-attention`).

`TinyDSA`'s main attention and output projection are stand-ins not compared against a shipped model. The lower-index tie-breaking rule is specific to this track because `torch.topk` does not guarantee an order for equal scores.
