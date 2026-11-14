## The idea

`mx.load` on a safetensors file parses the header and returns. Each value of the dict
is an array with a shape, a dtype, and one pending operation, `Load`, that knows the
file and the byte range. On this machine (M1 Max, MLX 0.32.4):

| file | tensors | `mx.load` | memory after `mx.load` | `mx.eval` of everything |
|---|---|---|---|---|
| `granite-4.0-350m`, 705 MB, bfloat16 | 226 | 0.9 ms | 0 bytes | 108 ms, 672 MiB |
| `granite-4.2-3b-mlx-8bit`, 3.9 GB | 927 | 2.5 ms | 0 bytes | 587 ms, 3,711 MiB |

So a loader has two phases with very different costs. The first is bookkeeping on
names, shapes, and dtypes, which are all known without evaluating: match names, find
what is missing and what is unexpected, compare shapes, and raise if the file does not
fit. The second is the read. Transposing and casting belong to neither: `value.T` and
`value.astype(dtype)` add two nodes on top of the `Load`, and they are computed when
the read happens.

That ordering gives the specification's guarantees almost for free:

- **All or nothing** holds because the module is updated once, after every check.
- **Unexpected tensors are never read** because the only thing evaluated is
  `module.parameters()`. A tensor that was not put into the module is not part of
  that graph. When the dict returned by `mx.load` goes out of scope, its `Load` node
  is dropped with its bytes still on disk.
- **The parameters come back evaluated** because of the single `mx.eval` at the end.
  Without it the module works, but its first forward pass silently includes reading
  the file, and a truncated file fails in the middle of generation, not at load.

## Reference solution (MLX)

```python
def load_into(module, path, *, prefix="", transpose_linear=False, strict=True):
    tensors = mx.load(path)  # lazy: one pending Load per tensor
    current = dict(tree_flatten(module.parameters()))
    linear_weights = {
        f"{name}.weight" if name else "weight"
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
    }

    loaded, unexpected = {}, []
    for key, value in tensors.items():
        name = key[len(prefix) :] if key.startswith(prefix) else None
        if name not in current:
            unexpected.append(key)
            continue
        if transpose_linear and name in linear_weights:
            value = value.T
        if value.shape != current[name].shape:
            raise ValueError(f"{key}: shape {value.shape} does not fit {name}")
        loaded[name] = value.astype(current[name].dtype)
    missing = sorted(set(current) - set(loaded))
    unexpected.sort()
    if strict and (missing or unexpected):
        raise ValueError(f"missing parameters: {missing}; unexpected tensors: {unexpected}")

    module.update(tree_unflatten(list(loaded.items())))
    mx.eval(module.parameters())
    return missing, unexpected
```

`tree_flatten` and `tree_unflatten` are inverses on the module tree: a list index is a
path component like any other, so `layers.0.q_proj.weight` round-trips through
`{"layers": [{"q_proj": {"weight": ...}}]}`. `Module.update` replaces the leaves it is
given and keeps the rest, which is exactly `strict=False`. `named_modules()` yields
every submodule with its path (the module itself has the empty path), so the set of
Linear weights is built from types, not from guesses about names or shapes.

## Why cast at all

A bfloat16 tensor loads as an MLX array of dtype `mx.bfloat16`, and MLX computes with
it, slowly: a 2048 by 2048 matrix product takes 3.0 ms in float32, 3.4 ms in float16,
and 6.2 ms in bfloat16 on this GPU. The module's dtype is the decision of whoever
built the module; the loader's job is to respect it. The cast from bfloat16 to
float32 is always exact. The cast to float16 is exact for magnitudes float16 can hold
(bfloat16 has 8 significand bits, float16 has 11, but a much smaller exponent range:
values below about $6 \times 10^{-8}$ round to zero and values above 65,504 overflow),
which covers the weights of ordinary checkpoints.

## Common mistakes

- **Forgiving a missing parameter.** `if strict and unexpected:` (or
  `Module.update` on whatever matched) reports success for a file that lacks a layer.
  The layer keeps its random initialization and the model still emits plausible
  tokens.
- **Forgiving an unexpected tensor.** The mirror image. An unexpected name under
  `strict` is usually a mapping bug (a wrong prefix makes *every* tensor unexpected and
  every parameter missing), so dropping it hides the real error.
- **No shape check.** `Module.update` assigns whatever it is given, and
  `Module.load_weights(..., strict=False)` skips its shape comparison along with the
  key comparison. A `(16, 32)` weight replaced by a `(32, 16)` one fails later, in a
  matmul far from the loader; a transposed square weight never fails at all.
- **Keeping the checkpoint's dtype.** Without the cast a float16 module becomes
  bfloat16 or float32, with the speed or the memory that implies.
- **`key.lstrip(prefix)`.** `lstrip` takes a set of characters. For the prefix
  `"model."` it removes leading `m`, `o`, `d`, `e`, `l`, and `.`, so
  `model.embed.weight` becomes `bed.weight` and `model.layers.0...` becomes
  `ayers.0...`. Slice after `startswith`, or use `str.removeprefix`.
- **Transposing every matrix.** An embedding table is 2-D and is not stored
  transposed. With a vocabulary as large as the width the shapes even agree.
- **Ignoring `transpose_linear`.** Rectangular weights then fail the shape check, and
  square ones (`q_proj`, `o_proj`) load transposed without a sound.
- **`mx.eval(tensors)` after `mx.load`.** Reads the whole file, including what the
  module does not use: 32 MiB instead of 0.1 MiB in the memory case.
- **No evaluation.** The module holds `Load`, `Transpose`, and `AsType` nodes, and the
  file handle, until somebody runs the model.
- **Updating while validating.** Writing each tensor into the module as it is matched
  leaves a half-loaded model behind when a later tensor raises.
- **Casting through numpy.** `mx.array(np.array(x))` evaluates the tensor, copies it
  to the host, and copies it back. numpy also has no bfloat16.

## Complexity

One pass over the names of the file and one over the parameters of the module, then a
single evaluation that reads exactly the bytes of the tensors that were loaded. Peak
memory is the loaded parameters plus, while the cast runs, the tensors in their file
dtype.

## In the wild

`mlx_lm.utils.load_model` is this function with model-specific renaming: it calls
`mx.load` on every shard, passes the dict through the model's `sanitize` method (which
drops and renames keys), calls `model.load_weights(list(weights.items()), strict=...)`,
and then `mx.eval(model.parameters())` unless it was called with `lazy=True`.
