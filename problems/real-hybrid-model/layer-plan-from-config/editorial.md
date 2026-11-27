## The idea

A model's `config.json` is a compact specification. When a model is instantiated, the reference configuration class expands it once into a concrete per-layer schedule, and the rest of the runtime consumes only that expanded plan. Porting a model accurately means reproducing that expansion. In Hugging Face Transformers 5.23, the logic lives in `__post_init__` of `Qwen3_5TextConfig` and `GlmMoeDsaConfig`, and in `__post_init__` plus `kv_sharing_modes` of `Gemma4TextConfig`.

Three patterns appear across all three families.

**Explicit lists take priority over rules.** Each family pairs a per-layer list (`layer_types`, `mlp_layer_types`, `indexer_types`) with scalar parameters that can generate it. The reference classes compute the fallback list only when the explicit list is `None`. Because the three published checkpoints include both the explicit list and matching scalars, an implementation that ignores the list passes on every published file and fails on the first hand-edited config. GLM adds a middle tier: `indexer_types`, then `index_topk_pattern` (such as `"FSSF"`), and finally the frequency-and-offset formula.

**Indexing and clamping rules differ by family.** Qwen3.5 counts layers from 1, placing `"full_attention"` at $(i + 1) \bmod n = 0$ where $n$ is `full_attention_interval`. With $n = 4$, full attention lands on layers 3, 7, 11, and so on. GLM's indexer rule uses $\max(i - o + 1, 0) \bmod \max(q, 1) = 0$, where $o$ is `index_skip_topk_offset` and $q$ is `index_topk_freq`. The lower clamp at 0 ensures that layers $0, \dots, o - 1$ all evaluate to 0 and run their own `"full"` indexer, followed by every $q$-th layer after that. With $o = 3$ and $q = 4$, full indexers land on layers 0, 1, 2, 6, 10, 14, and so on. Without the clamp, layer 0 would evaluate $-2 \bmod 4 = 2$ in Python and become `"shared"` with no earlier layer to share from. Clamping $q$ to at least 1 prevents division by zero and makes every layer `"full"` when $q = 0$.

**Defaults belong to the family, not to the parameter name.** A missing `partial_rotary_factor` defaults to `0.25` in Qwen3.5 (set explicitly in its config class under `# assign default for BC`), whereas Gemma 4's proportional RoPE defaults to `1.0`. A missing `index_skip_topk_offset` defaults to `2`, even though GLM-5.3 sets it to `3`. A missing `global_head_dim` defaults to `512`.

Gemma 4's KV sharing is a cross-layer dependency rather than a periodic schedule. When `num_kv_shared_layers` ($s$) is greater than 0, the last $s$ layers compute no keys or values. Instead, each reuses the KV tensors stored by the latest unshared layer of its own mixer type. That requires one `"store"` producer per distinct type. In Gemma-4-E2B-it, layer 13 stores for `"sliding_attention"` and layer 14 stores for `"full_attention"`. A single producer cannot serve both types because they use different head widths (`256` vs. `512`) and different RoPE settings (`"default"` with $\theta = 10^4$ vs. `"proportional"` with $\theta = 10^6$).

Head widths and rotated dimensions also follow three different conventions. Qwen3.5 has one `head_dim` and rotates `int(head_dim * f)` channels, where $f$ is `partial_rotary_factor`. GLM splits each head into a non-rotary part (`qk_nope_head_dim`) and a rotary part (`qk_rope_head_dim`), so the full query head width is their sum. GLM-5.3's top-level `head_dim` field is `192`, which is neither the sum (`256`) nor the rotary width (`64`). The config class overwrites `head_dim` with the rotary width on load. Gemma 4 uses `global_head_dim` for full-attention layers and `head_dim` for sliding-attention layers.

One edge case is intentionally excluded from the test suite. If `int(head_dim * f)` is odd in a Qwen3.5 config, Transformers still builds frequencies with `arange(0, dim, 2)`, which rotates one extra channel (for example, `head_dim = 100` and `f = 0.25` gives `int(25.0) = 25`, producing 13 frequencies and 26 rotated channels). All tests and published configs keep this value even.

## Reference solution (Python)

```python
def layer_plan(config):
    text = config.get("text_config", config)
    family = text.get("model_type")
    if family == "qwen3_5_text":
        return _qwen(text)
    if family == "glm_moe_dsa":
        return _glm(text)
    if family == "gemma4_text":
        return _gemma(text)
    raise ValueError(f"unsupported model_type {family!r}")


def _qwen(text):
    n = text["num_hidden_layers"]
    mixers = text.get("layer_types")
    if mixers is None:
        interval = text.get("full_attention_interval", 4)
        mixers = [FULL if (i + 1) % interval == 0 else LINEAR for i in range(n)]
    params = text["rope_parameters"]
    factor = params.get("partial_rotary_factor", 0.25)
    head = text["head_dim"]
    rope = _rope(params["rope_type"], params["rope_theta"], head, int(head * factor))
    return [_entry(mixer, rope=dict(rope) if mixer == FULL else None) for mixer in mixers]


def _glm(text):
    n = text["num_hidden_layers"]
    mlps = text.get("mlp_layer_types")
    if mlps is None:
        dense = text.get("first_k_dense_replace", 3)
        mlps = ["dense" if i < dense else "sparse" for i in range(n)]
    indexers = text.get("indexer_types")
    if indexers is None:
        pattern = text.get("index_topk_pattern")
        if pattern is not None:
            letters = {"F": "full", "S": "shared"}
            indexers = [letters[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
        else:
            freq = max(text.get("index_topk_freq", 1), 1)
            offset = text.get("index_skip_topk_offset", 2)
            indexers = ["full" if max(i - offset + 1, 0) % freq == 0 else "shared" for i in range(n)]
    params = text["rope_parameters"]
    rotary = text["qk_rope_head_dim"]
    rope = _rope(params["rope_type"], params["rope_theta"], text["qk_nope_head_dim"] + rotary, rotary)
    return [_entry("indexed_attention", mlps[i], indexers[i], dict(rope)) for i in range(n)]


def _gemma(text):
    kinds = text["layer_types"]
    n = len(kinds)
    shared = text.get("num_kv_shared_layers", 0)
    kv = [None] * n
    if shared:
        first = n - shared
        for i in range(first, n):
            kv[i] = "reuse"
        for kind in set(kinds):
            kv[max(i for i in range(first) if kinds[i] == kind)] = "store"
    plan = []
    for kind, mode in zip(kinds, kv, strict=True):
        params = text["rope_parameters"][kind]
        head = text.get("global_head_dim", 512) if kind == FULL else text["head_dim"]
        if params["rope_type"] == "proportional":
            rotary = 2 * int(params.get("partial_rotary_factor", 1.0) * head // 2)
        else:
            rotary = head
        plan.append(_entry(kind, rope=_rope(params["rope_type"], params["rope_theta"], head, rotary), kv=mode))
    return plan
```

`_entry` and `_rope` construct the two output dictionaries. Qwen3.5's linear-attention layers get `rope = None` because `Qwen3_5DecoderLayer` passes rotary embeddings only to its full-attention sublayer. A Gated DeltaNet layer tracks token order through its recurrence.

## Common mistakes

- **Counting Qwen3.5 layers from 0 (`i % interval == 0`).** This puts full attention at layers 0, 4, 8, ... instead of 3, 7, 11, .... Loading real weights fails loudly because `self_attn` and `linear_attn` keys mismatch, but config-only tools such as memory planners fail silently.
- **Ignoring the explicit per-layer list** or letting a pattern override it. This passes on every published file because the explicit lists and scalar rules agree there.
- **Ignoring `index_topk_pattern`.** GLM-5.3 sets this field to `null`, so the branch appears unused until a config populates it.
- **Dropping the clamp in GLM's indexer rule** (`(i - offset + 1) % freq`) or omitting the `+ 1` (`i - offset`). Dropping the clamp stops layers `0..offset - 1` from all being `"full"`. Omitting `+ 1` shifts every subsequent `"full"` layer by one.
- **Using `0` as the default for `index_skip_topk_offset`.** The reference default is `2`.
- **Using `1.0` as the default `partial_rotary_factor` for Qwen3.5.** While generic RoPE rotates all channels by default, Qwen3.5 defaults to `0.25`.
- **Reading GLM's `head_dim` directly.** In GLM configs, `head_dim` is only the non-rotary width, not `qk_nope_head_dim + qk_rope_head_dim`.
- **Marking only one `"store"` layer in Gemma 4.** Both the last unshared sliding-attention layer and the last unshared full-attention layer must be marked `"store"`.
- **Marking `"store"` layers when `num_kv_shared_layers == 0`.** When nothing is shared, `kv_sharing_modes` returns `None` for every layer.
- **Using one RoPE entry or one head width for all Gemma 4 layers.** Full-attention layers use `rope_parameters["full_attention"]` and `global_head_dim`, not the sliding-attention settings.
- **Using `int(f * head_dim)` for Gemma 4's proportional RoPE.** The reference implementation counts rotated pairs with `int(f * head_dim // 2)` and multiplies by 2, rounding odd products down to an even channel count (for example, `4` of `20` at `f = 0.25`, not `5`).
- **Attaching RoPE to Qwen3.5 linear-attention layers.** Those layers must have `rope = None`.
- **Importing `transformers` config classes.** The goal is to implement the expansion logic directly.

## Complexity

Time and space complexity are $O(L)$. Scanning for `"store"` layers in Gemma 4 takes $O(L \cdot |\text{types}|)$, where there are at most two mixer types.

## In the wild

The test fixtures were generated by instantiating `Qwen3_5TextConfig`, `GlmMoeDsaConfig`, and `Gemma4TextConfig` in Transformers 5.23.0 and reading their resolved schedules, `kv_sharing_modes`, per-layer head widths, and rotary-embedding frequency tables.

Two model behaviors are outside the scope of this plan. In Gemma 4, layers in `"reuse"` mode double their MLP width when `use_double_wide_mlp` is enabled. In GLM, the forward loop passes the top-$k$ indices from each `"full"` layer up the stack to subsequent `"shared"` layers, a cross-layer reuse technique described in "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse" (arXiv 2603.12201, abstract reviewed for this curriculum). The next two problems execute the Qwen3.5 plan on real weights. `real-gdn-block` runs one linear-attention layer, and `real-hybrid-logits` runs all 24 layers.
