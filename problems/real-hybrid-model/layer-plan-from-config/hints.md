Write down, for each family, which keys are *lists* (one entry per layer, to be copied)
and which are *rules* (a few numbers a list is generated from). Every list has a rule
behind it. Which do you consult first, and what do you do when the rule's own numbers
are missing too?

---

Unwrap first (`config.get("text_config", config)`), dispatch on `model_type`, and write one
small function per family that builds five lists of length $L$ and zips them into
dicts. Use `c.get(key)` and compare with `None` for the four list-or-pattern keys, and
`c.get(key, default)` for the numbers: 4 for `full_attention_interval`, 0.25 for
Qwen's `partial_rotary_factor`, 3 for `first_k_dense_replace`, 1 and 2 for
`index_topk_freq` and `index_skip_topk_offset`, 512 for `global_head_dim`, 0 for
`num_kv_shared_layers`.

Check the two schedules against the published files by hand before trusting them:
Qwen3.5's attention layers are 3, 7, 11, ..., and GLM-5.3's full indexers are 0, 1, 2,
6, 10, ....

---

The lines that go wrong:

```python
"full_attention" if (i + 1) % interval == 0 else "linear_attention"
"full" if max(i - offset + 1, 0) % max(freq, 1) == 0 else "shared"
```

For Gemma 4, with `first = L - shared` and only when `shared > 0`: mark
`range(first, L)` as `"reuse"`, then for each distinct layer type find the largest
`i < first` with `layer_types[i] == that type` and mark it `"store"`. The RoPE entry
and the head width are looked up per layer by the layer's own type; the proportional
type rotates `2 * int(f * head_dim // 2)` channels, which is not `int(f * head_dim)`
when that product is odd.
