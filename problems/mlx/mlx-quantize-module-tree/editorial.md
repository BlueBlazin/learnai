## The idea

An `nn.Module` is a dict. Its entries are arrays (parameters), other modules, and
lists and dicts of either, nested as deep as the author liked. A transformer keeps its
blocks in a list, so almost all of its weights are behind a container, and any walk
that only follows attributes that are modules sees the embedding, the head, and
nothing in between.

MLX gives you the walk in three pieces:

- `model.leaf_modules()` returns the modules without children as a nested structure
  that mirrors the tree: `{"embed": Embedding, "layers": [{"q_proj": Linear, ...}, ...]}`.
  Linear and embedding layers have no children, so every one of them is a leaf.
- `tree_map_with_path(fn, tree, is_leaf=nn.Module.is_module)` calls `fn(path, leaf)`
  for every leaf, where `path` joins the dict keys and list indices with dots, and
  returns the same structure built from the results.
- `model.update_modules(tree)` writes such a structure back into the model, replacing
  each module it names.

Quantizing a model is then one function from module to module, applied at every leaf.

The last line matters as much as the walk. `to_quantized` calls `mx.quantize`, which
returns three lazy arrays: nodes whose pending computation references the float
weight. After `update_modules` the float layers are gone from the tree, but not from
memory: the graph keeps every one of them alive until the codes are computed.
`mx.eval(model.parameters())` computes them, the graph is dropped, and the float
weights are freed.

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

`nn.QuantizedLinear` is not a subclass of `nn.Linear`, so the `isinstance` test also
keeps the walk away from layers that are already quantized. `convert` returns the
module itself for everything it does not replace, which is why untouched layers stay
the very same objects.

## What the numbers are

Measured on this machine (M1 Max, MLX 0.32.4) on an `MLXTinyLM` of width 512 with two
layers and a vocabulary of 1024: 6.3 million weights, 24.0 MiB in float32.

| recipe | bits per weight | parameter bytes | relative error of the logits |
|---|---|---|---|
| float32 | 32.0 | 24.0 MiB | 0 |
| 8 bits, groups of 64 | 9.01 | 6.76 MiB | 0.017 |
| 8 bits for `embed` and `head`, 4 bits elsewhere | 5.68 | 4.26 MiB | 0.24 |
| 4 bits, groups of 64 | 5.01 | 3.76 MiB | 0.29 |
| 4 bits, groups of 32 | 6.01 | 4.51 MiB | 0.30 |
| 2 bits, groups of 64 | 3.01 | 2.26 MiB | 0.90 |

Three things to read from it. The overhead is real: with float32 scales and biases, a
"4-bit" layer costs $4 + 2 \cdot 32 / 64 = 5$ bits per weight and an "8-bit" layer 9.
Checkpoints written by `mlx_lm.convert` keep scales and biases in 16-bit floats, which
gives the familiar 4.5 and 8.5. The numbers end in `.01` because the norm gains stay
in float32. And the error column is this model's, a random-weight toy whose weights
are far harder to quantize than a trained model's; it is there to show the direction
of each trade, not its size.

Left lazy, the 4-bit model holds 24.0 MiB when `quantize_model` returns, not 3.8: the
whole float model, kept alive by sixteen pending `Quantize` nodes.

## Common mistakes

- **Skipping children inside lists and dicts.** `for name, child in
  module.children().items(): if isinstance(child, nn.Module): ...` follows attributes
  and nothing else. On a transformer it reaches two layers out of sixteen, and the
  model is 14 percent smaller where it should be 84.
- **Forgetting the embedding.** Testing only for `nn.Linear` leaves the embedding in
  float32. In a small model that is a large share of the weights, and a recipe that
  asked for 8 bits there was silently ignored.
- **Treating the recipe's answer as yes or no.** `if recipe(path, module):
  module.to_quantized()` quantizes the right layers with the default 4 bits in groups
  of 64, whatever the dict said. A mixed recipe becomes a uniform one.
- **Quantizing what the recipe declined.** `recipe(path, module) or defaults` turns
  `None`, which means "leave this layer alone", into "use the defaults". The layers
  that were excluded on purpose, because they are the sensitive ones, get the
  coarsest setting.
- **Passing the leaf's name and not its path.** With `"q_proj"` for
  `"layers.0.q_proj"`, a recipe keyed on paths matches nothing, or matches the same
  layer of every block at once, and the returned record cannot be used to rebuild the
  tree.
- **Leaving the result lazy.** Everything looks right and the model is as large as
  before. The first forward pass then pays for the quantization of every layer it
  touches.
- **Bits per weight without the scales and biases.** Counting only the packed codes
  gives 4.0 where the model occupies 5.0: off by a fifth, in the direction that makes
  a model look as if it fits.
- **Dividing by stored elements.** `sum(p.size for p in parameters)` counts a uint32
  word as one weight when it holds eight, and counts scales and biases as weights.
  Every stored element is then four bytes, so a 4-bit model reports exactly 32 bits
  per weight, the same as the float model.
- **Calling `nn.quantize`.** It is this function, minus the record and the
  evaluation; see below.

## Complexity

One visit per module and one call of the recipe per linear or embedding layer.
Quantization is a pass over each selected weight. The peak memory of the call is the
float model plus the quantized one, because all layers are evaluated together;
evaluating layer by layer (`mx.eval` on each new layer's parameters as it is made)
keeps the peak near the float model plus one layer, at the price of one evaluation
per layer.

## In the wild

`nn.quantize(model, group_size, bits, class_predicate=...)` is the same walk: the
predicate receives `(path, module)` for every leaf module and returns `False`, `True`,
or a dict of arguments for `to_quantized`. `mlx_lm.utils.quantize_model` wraps it with
a predicate that skips layers whose width is not a multiple of the group size and
writes each dict answer into `config["quantization"][path]`; when a quantized model is
loaded, that record is turned back into a predicate so that the empty tree has the
right layer types before `load_weights` runs. `mlx_lm.utils.compute_bits_per_weight`
is the bits-per-weight count of this problem, with one difference: it counts weights
over leaf modules only, so a parameter that sits directly on a container module is
counted in the bytes and not in the weights.
