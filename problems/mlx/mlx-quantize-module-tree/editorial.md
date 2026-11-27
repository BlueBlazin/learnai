## The idea

An `nn.Module` stores its parameters, child modules, and arbitrarily nested lists or dicts of either. Because a transformer keeps its blocks inside a list, almost all of its weights sit behind a container. A walk that only inspects direct module attributes finds the embedding and output head and misses every block in between.

MLX provides three helpers that handle arbitrary nesting for you:

- `model.leaf_modules()` returns all modules that have no children, arranged in a nested dict-and-list structure that mirrors the model tree (for example, `{"embed": Embedding, "layers": [{"q_proj": Linear, ...}, ...]}`). Because `nn.Linear` and `nn.Embedding` have no child modules, every one of them is a leaf.
- `tree_map_with_path(fn, tree, is_leaf=nn.Module.is_module)` calls `fn(path, leaf)` on each leaf, where `path` joins dict keys and list indices with dots, and returns a new tree with the same structure holding the return values.
- `model.update_modules(tree)` writes that structure back into `model`, replacing each named module in place.

With those three helpers, quantization reduces to a single leaf-level mapping function.

Evaluating the result is just as important as the tree walk. Calling `to_quantized` runs `mx.quantize`, which returns three lazy arrays whose graph nodes still reference the original float weight. After `update_modules`, the float layers are no longer attached to the model, but the pending graph keeps every float weight alive in memory until the quantized arrays are computed. Calling `mx.eval(model.parameters())` materializes the quantized arrays, drops the graph, and frees the float weights.

## Reference solution (MLX)

```python
def quantize_model(model, recipe):
    quantized = {}

    def convert(path, module):
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return module
        params = recipe(path, module)
        if params is None:
            return module
        bits, group_size = params["bits"], params["group_size"]
        quantized[path] = {"bits": bits, "group_size": group_size}
        return module.to_quantized(group_size=group_size, bits=bits)

    leaves = tree_map_with_path(convert, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    mx.eval(model.parameters())
    return quantized


def bits_per_weight(model):
    arrays = [value for _, value in tree_flatten(model.parameters())]
    stored_bytes = sum(value.nbytes for value in arrays)
    weights = sum(value.size for value in arrays)
    for _, module in model.named_modules():
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            stored = module.weight.size + module.scales.size + module.biases.size
            weights += module.weight.size * 32 // module.bits - stored
    return 8 * stored_bytes / weights
```

`nn.QuantizedLinear` is not a subclass of `nn.Linear` (and `nn.QuantizedEmbedding` is not a subclass of `nn.Embedding`), so the `isinstance` check automatically skips layers that are already quantized. Returning `module` unchanged for every unquantized leaf guarantees that untouched layers remain the exact same objects.

## What the numbers are

Measured on an M1 Max with MLX 0.32.4 on an `MLXTinyLM` of width 512, 2 layers, and vocabulary size 1024 (6.3 million weights, 24.0 MiB in float32):

| recipe | bits per weight | parameter bytes | relative error of the logits |
|---|---|---|---|
| float32 | 32.0 | 24.0 MiB | 0 |
| 8 bits, groups of 64 | 9.01 | 6.76 MiB | 0.017 |
| 8 bits for `embed` and `head`, 4 bits elsewhere | 5.68 | 4.26 MiB | 0.24 |
| 4 bits, groups of 64 | 5.01 | 3.76 MiB | 0.29 |
| 4 bits, groups of 32 | 6.01 | 4.51 MiB | 0.30 |
| 2 bits, groups of 64 | 3.01 | 2.26 MiB | 0.90 |

Three patterns stand out. First, scale and bias overhead is substantial. With float32 scales and biases, a 4-bit layer costs $4 + 2 \cdot 32 / 64 = 5$ bits per weight and an 8-bit layer costs 9. Checkpoints written by `mlx_lm.convert` store scales and biases in 16-bit floats, giving the familiar 4.5 and 8.5 bits per weight. Second, the `.01` tail comes from the norm gains staying in float32. Third, the logit error column is from a random-weight toy model whose weights are much harder to quantize than a trained model's, so it illustrates the direction of each trade-off rather than typical production accuracy.

If the model is left lazy, the 4-bit model still holds 24.0 MiB when `quantize_model` returns instead of 3.8 MiB, because sixteen pending `Quantize` nodes keep the float weights alive.

## Common mistakes

- **Skipping children inside lists and dicts.** Iterating over `module.children().items()` and checking `isinstance(child, nn.Module)` only follows direct attributes. On a transformer, it visits 2 layers out of 16 and shrinks the model by 14 percent instead of 84 percent.
- **Forgetting the embedding.** Checking only for `nn.Linear` leaves `nn.Embedding` in float32. In small models the embedding is a large fraction of the weights, and any recipe asking for 8-bit embeddings is silently ignored.
- **Treating the recipe's return value as a boolean.** Running `if recipe(path, module): module.to_quantized()` quantizes the right layers using the default 4 bits and group size 64, ignoring the dict values and turning a mixed recipe into a uniform one.
- **Quantizing layers the recipe declined.** Writing `recipe(path, module) or defaults` turns `None` into the default quantization settings, quantizing the sensitive layers that were intentionally excluded.
- **Passing the leaf name instead of its full path.** Passing `"q_proj"` instead of `"layers.0.q_proj"` breaks path-keyed recipes and produces a returned dict that cannot rebuild the module tree.
- **Leaving the result lazy.** Without `mx.eval(model.parameters())`, the model stays as large as the float model in memory, and the first forward pass pays the cost of quantizing every layer it touches.
- **Omitting scales and biases from bits per weight.** Counting only the packed codes reports 4.0 bits per weight when the model actually uses 5.0, understating memory usage by 20 percent.
- **Dividing by stored elements.** Using `sum(p.size for p in parameters)` counts each 4-byte `uint32` word as one weight instead of `32 / bits` weights and counts scales and biases as weights. Every stored element is then 4 bytes, so a 4-bit model reports 32.0 bits per weight.
- **Calling `nn.quantize`.** That built-in performs this exact walk (without recording the dict or evaluating), and is forbidden by the problem rules.

## Complexity

The walk visits each module once and calls `recipe` once per linear or embedding layer. Quantization makes a single pass over each selected weight tensor. Evaluating all parameters together gives a peak memory of roughly the float model plus the quantized model. Evaluating each newly quantized layer immediately with `mx.eval` lowers peak memory to roughly the float model plus one layer, at the cost of one evaluation per layer.

## In the wild

`nn.quantize(model, group_size, bits, class_predicate=...)` uses this exact traversal. Its predicate receives `(path, module)` for each leaf module and returns `False`, `True`, or a dict of arguments for `to_quantized`. `mlx_lm.utils.quantize_model` wraps `nn.quantize` with a predicate that skips layers whose width is not divisible by `group_size` and writes each dict result into `config["quantization"][path]`. When loading a quantized checkpoint, that record is converted back into a predicate so the empty model tree gets the matching quantized layer types before `load_weights` runs. `mlx_lm.utils.compute_bits_per_weight` computes the same metric as `bits_per_weight`, except it sums logical weights over leaf modules only, so a parameter attached directly to a container module is counted in `bytes` but not in `weights`.
