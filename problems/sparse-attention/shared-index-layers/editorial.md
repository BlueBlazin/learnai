## The idea

In a sparse-attention stack, every layer asks which past tokens matter to each query, and the answer changes little between adjacent layers. Index sharing exploits that stability. A subset of *full* layers run their own indexers, while every *shared* layer reuses the selection from the nearest full layer below it at zero indexer cost.

The stack therefore passes two pieces of state upward from layer to layer:

- The hidden state `h`, updated by every layer.
- The selection `idx`, updated by full layers only.

Three details determine whether the implementation matches the model:

1. **A full layer indexes its own input.** `dsa.indexer(i, h, k)` runs on the `h` entering layer `i`, after all lower layers have added their residuals. Selections cannot be precomputed before the loop.
2. **A shared layer reuses indices, not an indexer.** It does not rerun the nearest full layer's indexer on its own `h`. It attends with the integer tensor that full layer produced, even though `h` has changed. In the shipped model, shared layers have no indexer weights at all.
3. **Nearest, not first.** Each full layer starts a new sharing group. Keeping a single `idx` variable that full layers overwrite gives "nearest" automatically.

The error case follows from the same `idx` variable. If a shared layer runs while `idx` is still `None`, no full layer has produced a selection yet, so the function raises `ValueError`.

## Reference solution (PyTorch)

```python
def forward_shared(dsa, h, types, k):
    idx = None
    for layer, kind in enumerate(types):
        if kind == "full":
            idx = dsa.indexer(layer, h, k)
        elif idx is None:
            raise ValueError(f"layer {layer} is shared, but no full layer comes before it")
        h = h + dsa.attend(layer, h, idx)
    return h
```

The JAX and MLX solutions are identical apart from type annotations because all tensor work except the residual addition happens inside the kit.

## Common mistakes

- **Running an indexer at every layer.** Ignoring `types` produces a different function and incurs one indexer call per layer.
- **Indexing in a shared layer and discarding the result.** The output matches, but the call count matches the unshared model.
- **Reusing the first full layer's indices everywhere.** This works when there is only one full layer, then fails from the second full layer onward.
- **Rerunning the nearest full layer's indexer.** Calling `dsa.indexer(nearest, h, k)` on the shared layer's `h` selects different tokens because `h` has changed, and adds an extra indexer call per shared layer.
- **Hoisting selections before the loop.** Computing every full layer's selection from the initial input `h` is right only for layer 0.
- **Reading `dsa.types` instead of the `types` argument.** `dsa.types` is only a default pattern on the object, not the configuration passed to `forward_shared`.
- **Forgetting to pass `k`.** Omitting `k` makes the indexer use its default selection size.
- **Dropping the residual or updating `h` in place.** Writing `h += ...` in PyTorch mutates the caller's input tensor.
- **Mishandling a leading shared layer.** Falling back to dense attention hides a misconfigured stack, and leaving `idx` undefined raises `UnboundLocalError` instead of `ValueError`.
- **Skipping the `idx is None` check.** Passing `None` to `dsa.attend` only raises `ValueError` by coincidence inside the kit, which is why tests require the error before any kit call. A real attention kernel handed no indices either attends densely or reads stale buffer contents, so the layer must check first.
- **Tampering with `indexer_calls` or using `with_framework`.** Rejected by the harness.
- **Calling `dsa.forward_ref`.** The kit's oracle is rejected by the harness.

## Complexity

With $F$ full layers out of $L$ total layers, indexer work drops from $L$ layers to $F$ layers while attention work stays the same. Each indexer call computes one indexer key per token, so `index_key_rows` is $F \cdot T$. Memory usage is unchanged during a single pass, and during cached decoding (`indexer-cache-decode`) shared layers need no indexer key cache at all.

## In the wild

Checked on 2026-11-21.

- **Implementation.** In `transformers` 5.23.0 (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`), `GlmMoeDsaAttention.__init__` builds an indexer only when `config.indexer_types[layer_idx]` is not `"shared"`. Its `forward` method uses `prev_topk_indices` on a shared layer and raises `ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")` if none exist. Every layer returns the indices it used, and `GlmMoeDsaModel.forward` passes them to the next layer. The residual addition lives in `GlmMoeDsaDecoderLayer`.
- **Verified execution.** `fixtures/make.py` builds a small `GlmMoeDsaModel` with the layer types of GLM-5.3's first 11 layers, wraps every indexer and attention module, and records which indexer produced each layer's indices. Layers 0, 1, 2, 6, and 10 run their own indexers (5 calls in total), while layers 3–5 reuse layer 2's selection and layers 7–9 reuse layer 6's selection. The public test for GLM-5.3's first 11 layers compares your pattern against that trace, and the same script confirms the `ValueError` when layer 0 is shared.
- **Configuration.** `zai-org/GLM-5.3` (`config.json`) sets `indexer_types` across its 78 layers to `"full"` for layers 0, 1, and 2, followed by a repeating pattern of three `"shared"` layers and one `"full"` layer (21 full layers in total). That schedule matches `GlmMoeDsaConfig`'s default rule with `index_topk_freq = 4` and `index_skip_topk_offset = 3`, where layer `i` is full when `max(i - offset + 1, 0) % freq == 0`.
- **Paper.** *IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse* (arXiv 2603.12201) describes "Full layers that run their own indexers" and "Shared layers that simply reuse the nearest Full layer's top-k indices" in its abstract (only the abstract was read).

What this problem leaves out: `TinyDSA` models each layer as attention plus a residual. A full decoder layer also normalizes its input before attention and follows attention with a normalized MLP or mixture-of-experts block with its own residual, neither of which modifies the selection indices. The shipped model also uses latent attention with rotary positions and per-call cached indices (`indexer-cache-decode`). Only the layer sharing pattern was compared against the shipped model, not `TinyDSA`'s attention or stack outputs.
