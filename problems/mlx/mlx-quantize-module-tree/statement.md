Quantize an MLX model in place according to a per-layer recipe that chooses either a quantization setting or floating point for each linear and embedding layer. Then compute the resulting model size in bits per weight.

## Specification

```python
def quantize_model(model, recipe): ...
def bits_per_weight(model): ...
```

### `quantize_model(model, recipe)`

`quantize_model` modifies `model` (an `nn.Module`) in place.

A module's children are the modules stored in its attributes, either directly or inside nested lists and dicts of any depth. The **path** of a module is the dotted chain of attribute names, list indices, and dict keys leading from `model` to that module, such as `"embed"`, `"layers.0.q_proj"`, or `"heads.b.proj"`. These match the paths returned by `model.named_modules()` and the tensor name prefixes in a checkpoint.

For every `nn.Linear` and `nn.Embedding` in the tree, call `recipe(path, module)` exactly once. It returns one of two values:

- `None` to leave the layer untouched as the exact same object.
- A dict `{"bits": b, "group_size": g}` to replace the layer at the same position in the tree with its affine-quantized counterpart. An `nn.Linear` becomes an `nn.QuantizedLinear` (keeping its bias if it has one), and an `nn.Embedding` becomes an `nn.QuantizedEmbedding`.

Do not call `recipe` on any other module, such as norms, containers, or layers that are already quantized. Leave all such modules unchanged. The traversal order is not specified.

Return a dict mapping the path of each replaced layer to its `{"bits": b, "group_size": g}` dict, and omit all unquantized layers. Converters write this per-layer record into `config.json` so a loader can reconstruct the same mixed-precision tree.

**Evaluate all parameters before returning.** `mx.quantize` is lazy, and its pending graph keeps references to the original float weights. Until the quantized arrays are evaluated, the model still holds the float weights in memory.

**Forbidden built-in.** Do not call `nn.quantize`. Everything else in `mlx.nn` and `mlx.utils` is allowed, including `Module.leaf_modules`, `Module.named_modules`, `Module.update_modules`, `tree_map_with_path`, and the layers' `to_quantized` methods.

### `bits_per_weight(model)`

`bits_per_weight` returns the average storage cost per logical weight as a Python `float`:

$$
\text{bits per weight} = \frac{8 \cdot \text{bytes}}{\text{weights}} .
$$

Here `bytes` is the total byte size of every parameter array in `model.parameters()`, including packed codes, scales, biases, float weights, and norm gains.

`weights` is the total number of logical weights represented by those arrays. Every element of a parameter array counts as one weight, except inside an `nn.QuantizedLinear` or `nn.QuantizedEmbedding`. In those quantized layers, the packed `uint32` array `weight` encodes `32 / bits` weights per element (totaling `out * in` weights), while `scales` and `biases` count as zero weights because they are storage overhead rather than model weights.

Quantizing a model therefore changes `bytes` while keeping `weights` unchanged.

## Shapes

| name | type |
|---|---|
| `model` | `nn.Module` (the root is never itself an `nn.Linear` or `nn.Embedding`) |
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

Suppose `recipe` assigns 8 bits to `embed` and `head`, 4 bits to the block layers, and `None` (float32) to `blocks.1.mlp`, all with `group_size=64`:

```python
quantize_model(model, recipe)
# {"embed": {"bits": 8, "group_size": 64}, "head": {"bits": 8, "group_size": 64},
#  "blocks.0.attn": {"bits": 4, "group_size": 64}, "blocks.0.mlp": {"bits": 4, "group_size": 64},
#  "blocks.1.attn": {"bits": 4, "group_size": 64}}

bits_per_weight(model)  # 11.4568   (the float model: 32.0)
```

`recipe` is called six times (once each for `embed`, `blocks.0.attn`, `blocks.0.mlp`, `blocks.1.attn`, `blocks.1.mlp`, and `head`) and never for a norm.

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

Each group of 64 weights in a row carries one float32 scale and one float32 bias, so a 64 × 64 layer adds $2 \cdot 64 \cdot 4 = 512$ bytes of overhead. Overall, $8 \cdot 29696 / 20736 = 11.4568$ bits per weight. The single layer left in float32 accounts for a fifth of the weights and more than half of the bytes.

## Constraints

- No module appears twice in the tree. Every layer selected by `recipe` has an input width divisible by its `group_size`. Allowed values are `bits` in `{2, 4, 8}` and `group_size` in `{32, 64, 128}`.
- The quantized model is compared against the same float model quantized with `nn.quantize` using an equivalent `class_predicate`. Packed codes must match exactly, floating-point parameters must match within relative tolerance `1e-4`, and model outputs must match within relative tolerance `1e-4` and absolute tolerance `1e-4`.
- `bits_per_weight` must match within relative tolerance `1e-6`.
- `recipe` must be called exactly once per `nn.Linear` and `nn.Embedding`.
- All model parameters must have 0 pending operations on return, verified with `mx.export_to_dot`.
- **Size.** Parameter bytes must match what the recipe implies. For the public transformer test ([`MLXTinyLM`](../../../python/src/learnai/kit/mlxlm.py) of width 64, 8 bits for `embed` and `head`, 4 bits elsewhere), total parameter bytes must be at most 69,888 (down from 361,728). On a wider model with the same recipe, total bytes must be at most 0.2 of the float model's bytes.
- **Memory.** On the Apple GPU, a model with 24 MiB of float32 weights quantized to 4 bits must hold at most 8 MiB when `quantize_model` returns (the reference holds 3.8 MiB).

## Why this matters

Production quantized checkpoints rarely use a single bit width everywhere. Converters keep embeddings and output heads at higher precision than transformer blocks, leave sensitive layers in floating point, and skip layers whose width is not divisible by the group size. `mlx_lm.convert` takes a predicate of this exact form and saves its per-layer choices in `config.json` so the loader can rebuild the matching module tree before loading weights.

A tree walk that skips lists or dicts silently quantizes only the embedding and output head of a transformer while leaving every block in floating point. Likewise, bits per weight governs checkpoint size, memory footprint, and decode bandwidth. Omitting scales and biases reports 4.0 bits per weight for a model that actually takes 4.5 or 5.0, leading to out-of-memory failures on devices where the model was expected to fit.
