For each model family, separate the config keys into explicit per-layer lists (copied directly) and fallback rules (scalar parameters used to generate a list). Every explicit list has a fallback rule behind it. Decide which one takes precedence, and check what default each scalar parameter uses when it is also missing.

---

Start by unwrapping the text config with `config.get("text_config", config)`, then dispatch on `model_type` to one small helper function per family. Each helper can build five lists of length $L$ and zip them into dictionaries.

Use `c.get(key)` and check `is None` for the four list-or-pattern keys (`layer_types`, `mlp_layer_types`, `indexer_types`, `index_topk_pattern`). Use `c.get(key, default)` for the numeric defaults: `4` for `full_attention_interval`, `0.25` for Qwen3.5's `partial_rotary_factor`, `3` for `first_k_dense_replace`, `1` for `index_topk_freq`, `2` for `index_skip_topk_offset`, `512` for `global_head_dim`, and `0` for `num_kv_shared_layers`.

Verify both schedule formulas by hand against the published configs. Qwen3.5 places full attention at layers 3, 7, 11, ..., and GLM-5.3 places full indexers at layers 0, 1, 2, 6, 10, ....

---

The two schedule formulas that are easiest to get slightly wrong are:

```python
"full_attention" if (i + 1) % interval == 0 else "linear_attention"
"full" if max(i - offset + 1, 0) % max(freq, 1) == 0 else "shared"
```

For Gemma 4, when `shared > 0`, let `first = L - shared`. Mark `range(first, L)` as `"reuse"`, and for each distinct mixer type in `layer_types`, find the largest `i < first` with that type and mark it `"store"`. Look up both the RoPE parameters and the head width by each layer's own mixer type. For `"proportional"` RoPE, compute `2 * int(f * head_dim // 2)`, which differs from `int(f * head_dim)` whenever `int(f * head_dim)` is odd.
