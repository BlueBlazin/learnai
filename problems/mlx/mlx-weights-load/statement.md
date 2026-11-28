Load a safetensors checkpoint into the parameters of an `mlx.nn.Module`. You will match checkpoint tensor names to the module tree, fix tensor layouts and dtypes, and reject incompatible files without modifying the module. Because `mx.load` is lazy, you can validate everything before reading a single weight from disk and avoid ever reading unused tensors.

## Specification

```python
def load_into(module, path, *, prefix="", transpose_linear=False, strict=True): ...
```

`module` is any `mlx.nn.Module`. Its **parameters** are the arrays returned by `module.parameters()`, identified by their dotted paths from `mlx.utils.tree_flatten` (such as `embed.weight`, `layers.0.q_proj.weight`, or `heads.a.bias`).

`path` points to a safetensors file. Calling `mx.load(path)` returns a dict mapping tensor names to MLX arrays **without reading the tensor bytes**. Each array is a lazy graph node that reads from disk only when evaluated.

`load_into` updates the parameters of `module` in place and returns `(missing, unexpected)`:

1. **Names.** A checkpoint tensor named `prefix + p` maps to module parameter `p`, stripping `prefix` once from the start of the name. Any tensor whose name does not start with `prefix`, or whose stripped name is not a parameter path in `module`, is **unexpected**. Any module parameter with no matching checkpoint tensor is **missing**.
2. **Layout.** When `transpose_linear` is `True`, the checkpoint stores `nn.Linear` weights as `(in, out)` while MLX expects `(out, in)`, so you must transpose them. A linear weight is the `weight` parameter of any submodule that is an instance of `nn.Linear` at any nesting depth. Do not transpose any other 2D parameter, such as an embedding table.
3. **Shape.** After applying step 2, if a tensor's shape does not match its target parameter's shape, raise `ValueError` regardless of `strict`.
4. **Dtype.** Cast each matched tensor to the current dtype of its target parameter. Loading a `bfloat16` checkpoint into a `float32` module produces `float32` parameters, and loading into a `float16` module produces `float16` parameters.
5. **Strictness.** When `strict=True`, raise `ValueError` if any parameter is missing or any tensor is unexpected. When `strict=False`, ignore unexpected tensors and leave missing parameters unchanged.
6. **All or nothing.** If `load_into` raises `ValueError`, no parameter of `module` may be modified.
7. **Return value.** `missing` is the sorted list of missing parameter paths. `unexpected` is the sorted list of unexpected tensor names exactly as they appear in the file (including `prefix` if present). A successful `strict=True` load returns `([], [])`.

**Evaluation rules.** The judge inspects the computation graph and peak memory:

- On return, every parameter of `module` must be evaluated with no pending operations, so the first forward pass does not trigger disk I/O.
- Unexpected tensors must never be evaluated. Their bytes must remain on disk.
- Do not trigger implicit evaluation with `.item()`, `.tolist()`, `np.array(x)`, or `print(x)`. Perform transposes and casts with MLX operations.

## Shapes

| name | shape | dtype |
|---|---|---|
| parameter of `module` | any | `float32` or `float16` |
| tensor in file | target parameter's shape, or `(in, out)` for an `nn.Linear` weight when `transpose_linear=True` | `float32`, `float16`, or `bfloat16` |
| `missing`, `unexpected` (result) | lists of `str` | - |

## Example

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 2)  # embed.weight: (5, 2)
        self.out = nn.Linear(2, 3)  # out.weight: (3, 2), out.bias: (3,)

    def __call__(self, ids):
        return self.out(self.embed(ids))
```

Suppose `a.safetensors` contains three tensors:

| name | dtype | value |
|---|---|---|
| `model.embed.weight` | `bfloat16` | `[[0, 0], [1, 0], [0, 1], [1, 1], [2, -1]]` |
| `model.out.weight` | `float32` | `[[1, 2], [3, 4], [5, 6]]` |
| `model.out.bias` | `float32` | `[0.5, 0, -0.5]` |

```python
net = Net()
load_into(net, "a.safetensors", prefix="model.")  # ([], [])
net.embed.weight.dtype  # float32: cast from bfloat16
net(mx.array([1, 4]))  # [[1.5, 3.0, 4.5], [0.5, 2.0, 3.5]]
```

Token 4 embeds to $[2, -1]$, and the output layer computes $[1 \cdot 2 - 2,\; 3 \cdot 2 - 4,\; 5 \cdot 2 - 6] + [0.5, 0, -0.5] = [0.5, 2, 3.5]$.

Now suppose `b.safetensors` has `model.embed.weight` and `model.out.weight` as above, omits the bias, and adds an extra `(2, 2)` tensor named `model.vision.weight`:

```python
load_into(net, "b.safetensors", prefix="model.")
# ValueError; net is completely unchanged

load_into(net, "b.safetensors", prefix="model.", strict=False)
# (["out.bias"], ["model.vision.weight"])
# embed.weight and out.weight are loaded, out.bias keeps its value,
# and model.vision.weight is never read from disk
```

## Constraints

- Files contain at most a few dozen tensors in `float32`, `float16`, or `bfloat16`. Module parameters are `float32` or `float16`.
- Loaded parameters are checked **exactly** against the file values after casting (`float32` casts are exact, and `float16` casts round to nearest). Model outputs after loading are checked with relative tolerance `1e-4` and absolute tolerance `1e-5`.
- "Evaluated" means `mx.export_to_dot` shows no pending operations behind the array. Module parameters must have no pending operations, while unexpected tensors returned by `mx.load` must still retain their pending `Load`.
- Memory limit (Apple GPU): loading a small model with `strict=False` from a file that also contains an unexpected 32 MiB tensor may increase peak memory by at most 8 MiB. Reading that tensor costs 32 MiB, whereas skipping it costs under 1 MiB.
- `path` always points to a readable safetensors file.

> [!NOTE]
> `Module.load_weights(pairs, strict=False)` does not satisfy rule 5. With `strict=False`, it also disables shape checks and silently replaces parameters with mismatched shapes.

## Why this matters

Every MLX inference run begins with checkpoint loading, and raw checkpoints rarely match a model's exact layout. Hugging Face weights often carry a `model.` prefix that your module omits, GPT-2 checkpoints store linear weights transposed as `(in, out)`, and most modern checkpoints use `bfloat16` even though Apple GPUs run `float16` faster. Multimodal checkpoints also bundle extra components like vision towers that a text-only run never needs.

Small bugs in a loader fail silently and expensively. Ignoring a missing key leaves a layer randomly initialized while the model still generates fluent-looking text. Eagerly evaluating the entire checkpoint before filtering doubles both load time and peak memory on models that barely fit in unified memory.
