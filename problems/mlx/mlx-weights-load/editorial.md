## The idea

When `mx.load` opens a safetensors file, it parses only the JSON header and returns immediately. Each value in the returned dict knows its shape and dtype and holds a single pending `Load` operation pointing to its byte range on disk. Measured on an M1 Max (MLX 0.32.4):

| file | tensors | `mx.load` | memory after `mx.load` | `mx.eval` of everything |
|---|---|---|---|---|
| `granite-4.0-350m`, 705 MB, bfloat16 | 226 | 0.9 ms | 0 bytes | 108 ms, 672 MiB |
| `granite-4.2-3b-mlx-8bit`, 3.9 GB | 927 | 2.5 ms | 0 bytes | 587 ms, 3,711 MiB |

This splits checkpoint loading into two distinct phases. First, validate names, shapes, and dtypes purely from metadata without reading any tensor bytes. Match names, collect missing and unexpected keys, check shapes, and raise on any error. Calling `value.T` and `value.astype(dtype)` during this phase only appends lazy graph nodes on top of each `Load` node. Second, apply the validated tree to the module and run a single `mx.eval(module.parameters())` to execute the reads, transposes, and casts together.

This two-phase structure delivers every contract guarantee cleanly:

- **All or nothing.** `module.update` runs only after every check succeeds.
- **Unused tensors stay on disk.** Only `module.parameters()` is evaluated. Unexpected tensors never enter the module tree, so their `Load` nodes are dropped when the dict from `mx.load` goes out of scope.
- **Eagerly materialized parameters.** The final `mx.eval` ensures all loaded weights are in memory before `load_into` returns, so the first forward pass does not absorb file I/O and a truncated file fails immediately at load time.

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

`tree_flatten` and `tree_unflatten` are exact inverses on a module's parameter tree. List indices act as normal path segments, so `layers.0.q_proj.weight` round-trips through `{"layers": [{"q_proj": {"weight": ...}}]}`. `Module.update` replaces only the leaves present in the tree you pass and keeps all other parameters intact, which matches `strict=False`. Using `module.named_modules()` identifies `nn.Linear` weights by submodule type rather than guessing from tensor names or shapes (the root module itself has the empty string `""` as its name).

## Why cast at all

MLX loads `bfloat16` tensors natively, but Apple GPUs execute `bfloat16` slower than `float16` or `float32`. A 2048 × 2048 matrix multiplication takes 3.0 ms in `float32`, 3.4 ms in `float16`, and 6.2 ms in `bfloat16` on an M1 Max. The module's existing dtype reflects the precision the caller chose, so the loader must preserve it. Casting `bfloat16` to `float32` is always exact. Casting `bfloat16` (8 significand bits) to `float16` (11 significand bits) is exact for ordinary model weights, though values smaller than roughly $6 \times 10^{-8}$ underflow to zero and values above 65,504 overflow.

## Common mistakes

- **Forgiving a missing parameter.** Checking only `if strict and unexpected:` (or calling `Module.update` directly on matched keys) silently accepts checkpoints missing a layer. The missing layer keeps its random initialization, and the model still outputs fluent-looking tokens.
- **Forgiving an unexpected tensor.** Under `strict=True`, unexpected keys usually indicate a naming or prefix mismatch (for example, a wrong prefix marks every tensor unexpected and every parameter missing). Ignoring them hides the real error.
- **Skipping the shape check.** `Module.update` assigns any array without checking shape, and `Module.load_weights(..., strict=False)` disables shape validation alongside key validation. A `(16, 32)` weight replaced by `(32, 16)` fails later in a distant matmul, and a transposed square weight never raises at all.
- **Keeping the checkpoint's dtype.** Skipping `.astype(current[name].dtype)` silently turns a `float16` module into `bfloat16` or `float32`, hurting speed or memory.
- **Using `key.lstrip(prefix)`.** `str.lstrip` strips a *set of characters*, not a literal prefix. With `prefix="model."`, it strips leading `m`, `o`, `d`, `e`, `l`, and `.`, turning `model.embed.weight` into `bed.weight` and `model.layers.0...` into `ayers.0...`. Check `key.startswith(prefix)` and slice, or use `str.removeprefix`.
- **Transposing every 2D matrix.** Embedding tables are 2D matrices and are not stored transposed. If vocabulary size happens to equal embedding width, the shapes even match silently.
- **Ignoring `transpose_linear`.** Rectangular linear weights fail the shape check, while square weights (`q_proj`, `o_proj`) load transposed without raising an error.
- **Calling `mx.eval(tensors)` right after `mx.load`.** This forces MLX to read the entire file, including unused tensors—costing 32 MiB instead of 0.1 MiB in the memory test.
- **Omitting the final `mx.eval`.** Leaving pending `Load`, `Transpose`, and `AsType` nodes in the module keeps the file handle open and delays disk reads until the first forward pass.
- **Updating while validating.** Mutating `module` inside the validation loop leaves a half-loaded model behind if a later tensor raises `ValueError`.
- **Casting through NumPy.** Calling `mx.array(np.array(x))` evaluates the array, copies it to host memory, and copies it back. NumPy also has no `bfloat16` dtype.

## Complexity

Validation makes one pass over the checkpoint keys and one pass over the module parameters. The single `mx.eval` reads only the bytes of matched tensors. Peak memory is the size of the loaded parameters plus temporary space for their file-dtype buffers during casting.

## In the wild

`mlx_lm.utils.load_model` follows this exact pattern with model-specific key remapping. It calls `mx.load` on every shard, passes the dict through the model's `sanitize` method to rename or drop keys, calls `model.load_weights(list(weights.items()), strict=...)`, and evaluates `model.parameters()` unless `lazy=True` is requested.
