Routing gives each token a list of `k` experts and `k` weights. Invert that view so that each expert has a list of the tokens that chose it along with their weights. Once you have those lists, each active expert takes three steps: gather (`x[tokens]`), run the expert once, and scatter-add the weighted outputs back to `tokens`. How can you build all the expert lists without looping over tokens and without boolean-mask indexing (which MLX does not support)?

---

Flatten the `(N, k)` routing tensors (`chosen` and `weights`) into `N * k` assignments: `flat_expert = chosen.reshape(-1)`, `flat_token = repeat(arange(N), k)`, and `flat_weight = weights.reshape(-1)`. Sort the assignments by expert index using `order = argsort(flat_expert)`. In this sorted order, each expert occupies one contiguous run of `order`, and the run lengths are the counts of each expert in `flat_expert` (computed with `bincount`). Convert the counts to Python ints so you can slice each run and skip any expert whose count is 0.

---

```
y = zeros_like(x)
start = 0
for expert, count in enumerate(counts):
    if count == 0:
        continue
    run = order[start : start + count]
    start += count
    tokens = flat_token[run]
    out = experts(expert, x[tokens])
    y[tokens] += flat_weight[run, None] * out      # scatter-add
```

The scatter must accumulate rather than overwrite, because each token receives `k` contributions, and it must index by `tokens` rather than a slice because an expert's tokens are spread across `x`. In PyTorch, use `y.index_add(0, tokens, ...)`. In JAX and MLX, use `y.at[tokens].add(...)`. Within a single expert's run, token indices are distinct (a token cannot pick the same expert twice), so indexed `+=` is safe as well. For the routing weights, taking a softmax over just the `k` selected logits equals taking the full softmax and renormalizing over those `k` entries.
