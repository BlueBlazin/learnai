Load a safetensors checkpoint into the parameters of an `mlx.nn` module: match file
names to the module tree, fix the layout and the dtype of each tensor, and refuse a
file that does not fit. `mx.load` is lazy, so all of that can be decided before a
single weight is read from disk, and the tensors the module does not use never have
to be read at all.

## Specification

```python
def load_into(module, path, *, prefix="", transpose_linear=False, strict=True): ...
```

`module` is any `mlx.nn.Module`. Its **parameters** are the arrays of
`module.parameters()`, each named by its dotted path in the tree, as
`mlx.utils.tree_flatten` spells it: `embed.weight`, `layers.0.q_proj.weight`,
`heads.a.bias`.

`path` names a safetensors file. `mx.load(path)` returns a dict from tensor name to
array **without reading the tensor data**: every array is a pending graph node that
reads its bytes from the file when it is evaluated.

`load_into` replaces parameters of `module` with tensors of the file, in place, and
returns `(missing, unexpected)`.

1. **Names.** The tensor named `prefix + p` belongs to the parameter with path `p`.
   The prefix is removed once, from the front. A tensor whose name does not start
   with `prefix`, or whose remainder is not a parameter path of `module`, is
   **unexpected**. A parameter that no tensor belongs to is **missing**.
2. **Layout.** If `transpose_linear` is true, the file stores the weight of every
   `nn.Linear` as `(in, out)`. MLX stores it as `(out, in)`, so each such tensor is
   transposed. A Linear weight is the parameter `weight` of a submodule that is an
   instance of `nn.Linear`, at any depth; nothing else is transposed (an embedding
   table is a matrix too).
3. **Shape.** After step 2, a tensor whose shape differs from the shape of its
   parameter raises `ValueError`, whatever `strict` is.
4. **Dtype.** Each tensor is cast to the dtype its parameter has now. A bfloat16
   file loaded into a float32 module gives float32 parameters; into a float16 module,
   float16 parameters.
5. **Strictness.** With `strict=True`, any missing parameter or unexpected tensor
   raises `ValueError`. With `strict=False`, unexpected tensors are ignored and
   missing parameters keep the values they have.
6. **All or nothing.** When `load_into` raises `ValueError`, no parameter of `module`
   has changed.
7. **Result.** `missing` is the sorted list of missing parameter paths; `unexpected`
   is the sorted list of unexpected tensor names, spelled as in the file (with the
   prefix, if they have it). Under `strict=True` a successful load returns `([], [])`.

**Evaluation rules.** The judge observes what is computed:

- On return, every parameter of `module` is evaluated: no operation is pending
  behind it, so the first forward pass does not pay for reading the file.
- An unexpected tensor is never evaluated. Its bytes stay on disk.
- Nothing is evaluated implicitly: no `.item()`, `.tolist()`, `np.array(x)`, or
  `print(x)` on an array. Casting and transposing are MLX operations.

## Shapes

| name | shape | dtype |
|---|---|---|
| a parameter of `module` | any | float32 or float16 |
| a tensor of the file | the parameter's shape; `(in, out)` for a Linear weight when `transpose_linear` is true | float32, float16, or bfloat16 |
| `missing`, `unexpected` (result) | lists of `str` | |

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

The file `a.safetensors` holds three tensors:

| name | dtype | value |
|---|---|---|
| `model.embed.weight` | bfloat16 | `[[0, 0], [1, 0], [0, 1], [1, 1], [2, -1]]` |
| `model.out.weight` | float32 | `[[1, 2], [3, 4], [5, 6]]` |
| `model.out.bias` | float32 | `[0.5, 0, -0.5]` |

```python
net = Net()
load_into(net, "a.safetensors", prefix="model.")  # ([], [])
net.embed.weight.dtype  # float32: cast from bfloat16
net(mx.array([1, 4]))  # [[1.5, 3.0, 4.5], [0.5, 2.0, 3.5]]
```

Token 4 embeds to $[2, -1]$, and the layer computes
$[1 \cdot 2 - 2,\; 3 \cdot 2 - 4,\; 5 \cdot 2 - 6] + [0.5, 0, -0.5] = [0.5, 2, 3.5]$.

The file `b.safetensors` has `model.embed.weight` and `model.out.weight` as above, no
bias, and a tensor `model.vision.weight` of shape `(2, 2)`:

```python
load_into(net, "b.safetensors", prefix="model.")
# ValueError; net is exactly as it was

load_into(net, "b.safetensors", prefix="model.", strict=False)
# (["out.bias"], ["model.vision.weight"])
# embed.weight and out.weight are loaded, out.bias keeps its value,
# and model.vision.weight was never read from the file
```

## Constraints

- Files hold at most a few dozen tensors, float32, float16, or bfloat16. Parameters
  are float32 or float16.
- Loaded parameters are compared **exactly** with the numbers in the file after the
  cast: a cast to float32 is exact, and a cast to float16 rounds to nearest. Model
  outputs after loading are compared to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`.
- "Evaluated" means that `mx.export_to_dot` lists no pending operation behind an
  array. The parameters must have none; an unexpected tensor returned by `mx.load`
  must still have its pending `Load`.
- Memory limit (on the Apple GPU): loading a small model from a file that also holds
  an unexpected 32 MiB tensor, with `strict=False`, may raise peak memory by at most
  8 MiB. Reading that tensor costs 32 MiB; not reading it, under 1 MiB.
- `path` always names a readable safetensors file.

> [!NOTE]
> `Module.load_weights(pairs, strict=False)` is not rule 5: with `strict=False` it also
> stops comparing shapes, and a tensor of the wrong shape silently replaces the
> parameter.

## Why this matters

Every model you run with MLX starts with this function, and checkpoints rarely fit as
they are. Hugging Face files name tensors `model.layers.0...` where your module says
`layers.0...`; GPT-2 stores its linear layers as `(in, out)`; nearly every recent
checkpoint is bfloat16, which Apple GPUs compute with more slowly than float16; and a
multimodal file carries a vision tower a text-only run never touches. A loader that
forgives a missing key gives you a model with a randomly initialized layer that still
produces fluent-looking tokens, the hardest kind of bug to notice. A loader that
evaluates the whole file to look at it doubles the load time and the peak memory of a
model that barely fits.
