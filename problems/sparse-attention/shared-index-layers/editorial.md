## The idea

In a sparse-attention stack the question "which tokens matter to this query?" is
asked again in every layer, and the answers barely change from one layer to the next.
Index sharing takes that at face value. A few layers, the *full* ones, ask the
question with their own indexer. Every other layer, a *shared* one, accepts the answer
of the nearest full layer below it and spends nothing on selection.

So the stack carries two pieces of state upward instead of one:

- the hidden state `h`, rewritten by every layer;
- the selection `idx`, rewritten by full layers only.

Three details decide whether an implementation computes the same function.

1. **A full layer selects from its own input.** `dsa.indexer(i, h, k)` uses the `h`
   that enters layer `i`, which already contains everything the layers below wrote.
   The selections cannot be computed ahead of the loop.
2. **A shared layer reuses indices, not an indexer.** It does not run the nearest
   full layer's indexer on its own hidden state. It attends with the integer tensor
   that layer produced, although `h` has moved on. In the shipped model a shared layer
   has no indexer weights at all, so there is nothing it could run.
3. **Nearest, not first.** Each full layer starts a new group. With one variable that
   full layers overwrite, "nearest" is what you get for free; getting "first" takes
   extra code.

The error case follows from the same variable: a shared layer that finds it unset
has nothing to attend with.

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

The JAX and MLX versions are identical apart from the type hints: all tensor
arithmetic except the residual add happens inside the kit.

## Common mistakes

- **Every layer indexes.** Ignoring `types` gives a model that is sparse in every
  layer by its own selection: a different function, and an indexer call per layer.
- **A shared layer indexes and discards.** The output is right and the cost is that
  of the unshared model. Only the call count shows it.
- **The first full layer's indices everywhere.** Right for any stack with a single
  group, wrong from the second full layer on.
- **The nearest full layer's indexer, rerun.** `dsa.indexer(nearest, h, k)` on the
  shared layer's input selects other tokens, because `h` has changed, and calls an
  indexer per layer.
- **Selections hoisted out of the loop,** computed from the stack's input. Right for
  layer 0 only.
- **The types of the object instead of the argument.** `dsa.types` is a default
  pattern; the stack to run is the one passed in.
- **`k` not passed on.** The indexer then selects its default number of tokens.
- **No residual,** or the residual in place. `h += ...` in PyTorch modifies the
  tensor the caller still holds.
- **The error case.** Attending to everything instead of raising hides a
  misconfigured stack. Leaving the variable unassigned raises `UnboundLocalError`
  from the middle of the loop instead of the `ValueError` the caller can handle.
- **No check at all.** Passing the unset selection on "works" here only because
  the kit's `attend` refuses `None` with a `ValueError` of its own, which is why the
  tests require the error before any kit call. A real attention kernel has no such
  courtesy: handed no indices it attends densely, or reads whatever the index
  buffer holds. The shipped layer makes the check itself for that reason.
- **Overwriting the call counter,** or running the skipped indexers on a
  `with_framework` twin. Rejected by the harness.
- **The kit's oracle.** `dsa.forward_ref` computes the answer and is rejected by the
  harness.

## Complexity

With $F$ full layers among $L$, the indexer work is $F$ times that of one layer
instead of $L$ times, and the attention work is unchanged. Per indexer call the kit
counts one indexer key per token, so `index_key_rows` is $F \cdot T$. Sharing changes
nothing in memory except that shared layers also need no indexer key cache
(`indexer-cache-decode`).

## In the wild

Checked on 2026-11-21.

- **The code.** `GlmMoeDsaAttention.__init__` in `transformers` 5.23.0
  (`models/glm_moe_dsa/modeling_glm_moe_dsa.py`) builds an indexer only when
  `config.indexer_types[layer_idx]` is not `"shared"`. Its `forward` uses
  `prev_topk_indices` on a shared layer and raises
  `ValueError("Shared DSA layers require top-k indices from a previous full indexer
  layer.")` when there are none; every layer returns the indices it used, and
  `GlmMoeDsaModel.forward` passes them to the next layer. The residual add is in
  `GlmMoeDsaDecoderLayer`.
- **Run, not only read.** `fixtures/make.py` builds a small `GlmMoeDsaModel` with the
  layer types of GLM-5.3's first 11 layers, wraps every indexer and attention, and
  records which indexer produced the indices each layer used: layers 0, 1, 2 their
  own, layers 3 to 5 those of layer 2, layer 6 its own, layers 7 to 9 those of layer
  6, layer 10 its own; 5 indexer calls. The public test "the first 11 layers of
  GLM-5.3" compares your pattern with that record. The same script confirms the
  `ValueError` for a shared first layer.
- **The configuration.** `zai-org/GLM-5.3` (`config.json`) lists `indexer_types` for
  its 78 layers: full for layers 0, 1, 2, then shared, shared, shared, full repeating,
  21 full layers in all. The same list follows from its `index_topk_freq` 4 and
  `index_skip_topk_offset` 3 by the default rule of `GlmMoeDsaConfig`, full when
  `max(i - offset + 1, 0) % freq == 0`.
- **The paper.** "IndexCache: Accelerating Sparse Attention via Cross-Layer Index
  Reuse" (arXiv 2603.12201). Its abstract describes "Full layers that run their own
  indexers" and "Shared layers that simply reuse the nearest Full layer's top-k
  indices". Only the abstract was read.

What this problem leaves out: `TinyDSA`'s layer is attention plus a residual. The
shipped decoder layer normalizes its input before the attention and follows it with a
normalized MLP or mixture-of-experts block, also with a residual; neither touches the
indices. The shipped attention is latent attention with rotary positions, and with a
cache the indices are per call (`indexer-cache-decode`). `TinyDSA`'s attention and
stack were not compared with a shipped model; the sharing pattern was.
