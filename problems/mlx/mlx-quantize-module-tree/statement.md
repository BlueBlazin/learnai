Quantize a whole MLX model in place, layer by layer, according to a recipe that says
for each layer how many bits it gets or that it stays in floating point, and then
report what the result costs in bits per weight. The quantized layers are MLX's own;
the work is walking a module tree whose layers sit in attributes, lists, and dicts,
replacing exactly the layers the recipe selects, and counting weights and bytes the way
the format stores them.

## Specification

```python
def quantize_model(model, recipe): ...
def bits_per_weight(model): ...
```

**`quantize_model(model, recipe)`** modifies `model`, an `nn.Module`, in place.

A module's children are the modules stored in its attributes, directly or inside
lists and dicts, nested to any depth. The **path** of a module is the dotted chain of
attribute names, list indices, and dict keys that leads to it from `model`:
`"embed"`, `"layers.0.q_proj"`, `"heads.b.proj"`. These are the paths that
`model.named_modules()` reports and the prefixes of the tensor names in a checkpoint.

For every module in the tree that is an `nn.Linear` or an `nn.Embedding`, call
`recipe(path, module)` exactly once. It returns

- `None`: the layer stays as it is, the very same object;
- a dict `{"bits": b, "group_size": g}`: the layer is replaced, at the same place in
  the tree, by its quantized twin in affine mode with those settings: an
  `nn.QuantizedLinear` for an `nn.Linear` (its bias, if any, is kept), an
  `nn.QuantizedEmbedding` for an `nn.Embedding`.

The recipe is not called for any other module: not for norms, not for containers, and
not for layers that are already quantized, which are left alone. The order of the
calls is not specified.

`quantize_model` returns a dict that maps the path of every layer it replaced to
`{"bits": b, "group_size": g}`, and nothing else. This is the per-layer record a
converter writes into `config.json` so that a loader can rebuild the same tree.

**On return, every parameter of the model is evaluated.** `mx.quantize` is lazy, and
its pending graph holds the float weight it was given; until the quantized arrays are
computed, the model occupies the memory of the float model.

**The rule of the exercise.** Do not call `nn.quantize`. Everything else in
`mlx.nn` and `mlx.utils` is allowed, including `Module.leaf_modules`,
`Module.named_modules`, `Module.update_modules`, `tree_map_with_path`, and the layers'
`to_quantized` methods.

**`bits_per_weight(model)`** returns, as a Python float,

$$
\text{bits per weight} = \frac{8 \cdot \text{bytes}}{\text{weights}} .
$$

- **bytes** is the total size in bytes of all parameter arrays of the model
  (`model.parameters()`): packed codes, scales, biases, float weights, norm gains,
  everything.
- **weights** is the number of weights those arrays stand for. Every element of a
  parameter array counts as one weight, with one exception: in an
  `nn.QuantizedLinear` or `nn.QuantizedEmbedding`, the packed `weight` counts as the
  `out * in` weights it encodes (`32 / bits` per uint32 word), and `scales` and
  `biases` count as none. They are overhead: bytes, but not weights.

So quantizing never changes the number of weights of a model, only its bytes.

## Shapes

| name | type |
|---|---|
| `model` | `nn.Module`; the root is never itself an `nn.Linear` or `nn.Embedding` |
| `recipe` | callable `(path: str, module: nn.Module) -> dict \| None` |
| result of `quantize_model` | `dict[str, dict[str, int]]` |
| result of `bits_per_weight` | Python `float` |

## Example

```python
class Block(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.attn = nn.Linear(dims, dims, bias=False)
        self.mlp = nn.Linear(dims, dims)  # with a bias
        self.norm = nn.RMSNorm(dims)


class Tiny(nn.Module):
    def __init__(self, vocab=32, dims=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dims)
        self.blocks = [Block(dims), Block(dims)]
        self.head = nn.Linear(dims, vocab, bias=False)
```

The recipe gives the embedding and the head 8 bits, the layers of the blocks 4 bits,
and keeps `blocks.1.mlp` in float32, all in groups of 64:

```python
quantize_model(model, recipe)
# {"embed": {"bits": 8, "group_size": 64}, "head": {"bits": 8, "group_size": 64},
#  "blocks.0.attn": {"bits": 4, "group_size": 64}, "blocks.0.mlp": {"bits": 4, "group_size": 64},
#  "blocks.1.attn": {"bits": 4, "group_size": 64}}

bits_per_weight(model)  # 11.4568   (the float model: 32.0)
```

The recipe is called six times, once for each of `embed`, `blocks.0.attn`,
`blocks.0.mlp`, `blocks.1.attn`, `blocks.1.mlp`, and `head`, and never for a norm.

| layer | weights | bytes after quantization |
|---|---|---|
| `embed`, 32 × 64 at 8 bits | 2048 | 2048 codes + 128 scales + 128 biases = 2304 |
| `blocks.0.attn`, 64 × 64 at 4 bits | 4096 | 2048 + 256 + 256 = 2560 |
| `blocks.0.mlp`, 64 × 64 at 4 bits, bias of 64 | 4160 | 2560 + 256 = 2816 |
| `blocks.0.norm` | 64 | 256 |
| `blocks.1.attn`, 64 × 64 at 4 bits | 4096 | 2560 |
| `blocks.1.mlp`, float32, bias of 64 | 4160 | 16640 |
| `blocks.1.norm` | 64 | 256 |
| `head`, 32 × 64 at 8 bits | 2048 | 2304 |
| total | 20736 | 29696 |

A scale and a bias are float32 here, one of each per row per group of 64, so a 64 × 64
layer carries $2 \cdot 64 \cdot 4 = 512$ bytes of them.
$8 \cdot 29696 / 20736 = 11.4568$ bits per weight; the one layer left in float32 holds
a fifth of the weights and more than half of the bytes.

## Constraints

- No module appears twice in the tree. Every layer the recipe selects has an input
  width that is a multiple of the group size it is given; `bits` is 2, 4, or 8 and
  `group_size` is 32, 64, or 128.
- The quantized model is compared with the same float model quantized by
  `nn.quantize` with an equivalent `class_predicate`: packed codes exactly, floating
  parameters to a relative tolerance of `1e-4`, and the model's output to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-4`.
- `bits_per_weight` is compared to a relative tolerance of `1e-6`.
- **Recipe calls** are counted exactly: once per `nn.Linear` and `nn.Embedding`.
- **Evaluation:** 0 pending operations behind the model's parameters on return
  (counted with `mx.export_to_dot`).
- **Size.** The parameter bytes are what the recipe implies. For the public
  transformer case (an [`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py) of
  width 64 with 8 bits for `embed` and `head` and 4 bits elsewhere): at most 69,888
  bytes, down from 361,728. For that recipe on a wider model: at most 0.2 of the float
  model's bytes.
- **Memory**, on the Apple GPU: a model with 24 MiB of float32 weights, quantized to
  4 bits, holds at most 8 MiB when `quantize_model` returns (the reference: 3.8 MiB).

## Why this matters

Nobody ships a model quantized uniformly. Converters keep the embedding and the output
head at more bits than the body, leave the layers that hurt most in higher precision,
and skip layers whose width does not fit the group size; `mlx_lm.convert` takes a
predicate of exactly this shape and records its answers per layer in `config.json`,
because the loader has to rebuild the same mixed tree before it can load a single
tensor. A walk that misses the layers inside a list quantizes the embedding and the
head of a transformer and none of its blocks. And "bits per weight" is the number
that the size of the file, the memory the model needs, and the speed of decoding all
follow from. Computed without the scales and biases it reads 4.0 for a model that
occupies 4.5 or 5 bits per weight, which is a model that does not fit in the memory
it was promised.
