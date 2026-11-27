## The idea

`mx.fast.metal_kernel` compiles a Metal source string into a GPU kernel. You provide the function body, and MLX generates the signature from the input/output names you declare and the built-in variables your body references. Passing `verbose=True` on the reference solution with float32 inputs prints the generated Metal code:

```cpp
template <typename T>
[[kernel]] void custom_kernel_fused_silu_mul__float_float_float_float(
  const device float* a [[buffer(0)]],
  const device float* b [[buffer(1)]],
  device float* out [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

    uint elem = thread_position_in_grid.x;
    T x = a[elem];
    out[elem] = x / (T(1) + metal::exp(-x)) * b[elem];
}
```

Every design choice follows directly from this signature.

- **The grid is the loop.** Metal invokes the kernel once for each coordinate in `grid`, using `thread_position_in_grid` as the thread index. An elementwise kernel needs one thread per element, so `grid=(a.size, 1, 1)` works for any shape. Because the output is always a row-major buffer, a flat element index is all each thread needs. `threadgroup` only controls how Metal groups threads for execution (at most 1024 threads per group; 256 is a solid default for a 1-D grid) and does not change the math.
- **Buffer pointers follow array dtypes, while local variables follow the template.** When `a` is float16, MLX binds `const device half* a`. Passing `template=[("T", a.dtype)]` runs the intermediate arithmetic in the input precision, and `output_dtypes=[a.dtype]` keeps the output at the same byte width.
- **Output buffers are not zero-initialized.** Any element not covered by a thread keeps whatever bytes the allocator handed out, either zeros from a fresh buffer or stale values from a recycled one.
- **0-D arrays are not buffers.** MLX passes a 0-D scalar as `const float& a`. Reshape 0-D inputs to `(1,)` before launching the kernel and reshape the output back to `()` afterward. Both reshapes are zero-copy.
- **Kernel calls are lazy.** Calling the kernel immediately returns a `CustomKernel` graph node. Metal compiles the kernel on its first execution, and MLX does not fuse custom kernels with neighboring operations, so build the kernel object once at module level and reuse it.

## Reference solution (MLX)

```python
_SOURCE = """
    uint elem = thread_position_in_grid.x;
    T x = a[elem];
    out[elem] = x / (T(1) + metal::exp(-x)) * b[elem];
"""

_KERNEL = mx.fast.metal_kernel(
    name="fused_silu_mul", input_names=["a", "b"], output_names=["out"], source=_SOURCE
)


def fused_silu_mul(a, b):
    shape = a.shape
    if a.ndim == 0:
        a, b = a.reshape(1), b.reshape(1)
    out = _KERNEL(
        inputs=[a, b],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    return out if out.shape == shape else out.reshape(shape)
```

Writing $\operatorname{silu}(x)$ as $x / (1 + e^{-x})$ avoids overflow across the entire input range. For large positive $x$, $e^{-x}$ underflows to `0` and the result is $x$. For large negative $x$, $e^{-x}$ overflows to `inf` in the denominator, giving the correct limit of `-0`. By contrast, writing the sigmoid as $e^x / (1 + e^x)$ evaluates to `inf / inf = NaN` as soon as $e^x$ overflows, which happens at $x = 11.09$ in float16.

## Strided inputs: copy or compute offsets

Transposing a `(64, 48)` array does not allocate a new buffer. Instead, `a.T` reuses the same memory with shape `(48, 64)` and strides `(1, 48)`, so element $(r, c)$ lives at offset $r \cdot 1 + c \cdot 48$.

With `ensure_row_contiguous=True` (the default), MLX copies strided inputs into row-major order before the kernel runs, so `a[elem]` is always valid. With `ensure_row_contiguous=False`, the kernel receives the raw buffer and must translate `elem` into a buffer offset itself. MLX provides a built-in helper and automatically adds `a_shape`, `a_strides`, and `a_ndim` to the signature when the body references them:

```cpp
T x = a[elem_to_loc(elem, a_shape, a_strides, a_ndim)];
```

`elem_to_loc` decomposes `elem` along each axis from last to first and accumulates `index * stride`. However, in MLX 0.32.4 it uses unsigned integer arithmetic, which breaks on reversed views (`a[::-1]`) because their negative strides wrap around and index out of bounds. To handle every view safely, perform the same loop in signed 64-bit integers:

```cpp
int64_t loc = 0;
uint rest = elem;
for (int d = a_ndim - 1; d >= 0; --d) {
    loc += int64_t(rest % uint(a_shape[d])) * a_strides[d];
    rest /= uint(a_shape[d]);
}
T x = a[loc];
```

Choosing between copying and strided indexing is a real performance trade-off. Measured on an M1 Max (MLX 0.32.4) using two transposed `2048 x 2048` float32 inputs (16 MiB each):

| kernel | peak memory above the inputs | time per call |
|---|---|---|
| default (MLX copies both views) | 48 MiB | 0.62 ms |
| strided (`ensure_row_contiguous=False`) | 16 MiB | 1.32 ms |
| contiguous inputs, either kernel | 16 MiB | 0.45 ms |

The contiguous copy is fast because it performs sequential writes, at the cost of allocating a temporary copy for each strided input. The strided kernel allocates only the output buffer, but pays for scattered memory reads and an integer division loop per element. Keep the default for typical activation kernels, and disable it only when large inputs are routinely strided (and always test reversed views when you do).

For comparison, the three built-in operations `mx.sigmoid(a) * a * b` take 0.47 ms on the same contiguous inputs. Hand-fusing a short elementwise chain saves little on a memory-bandwidth-bound GPU. Custom kernels pay off when MLX lacks the operation you need or when an unfused composition would materialize large intermediate tensors, which is the focus of the next problem.

## Common mistakes

- **Launching `a.shape[0]` threads** (or indexing a 1-D grid with `a.shape`). This passes 1-D vector tests, but on a `(64, 48)` matrix only 64 of the 3072 elements get written and the rest contain uninitialized memory.
- **Hardcoding `float` regardless of input dtype.** Passing `template=[("T", mx.float32)]` and `output_dtypes=[mx.float32]` still reads float16 inputs without error (since input pointer types follow the array), but silently promotes the output to float32, doubling memory usage and promoting downstream operations.
- **Setting `ensure_row_contiguous=False` while keeping `a[elem]`.** Contiguous arrays still pass, but transposed or sliced views are read in raw buffer order, scrambling the output.
- **Using `elem_to_loc` for reversed views.** It works for transposed, stepped, and broadcast views, but wraps around on negative strides.
- **Writing `exp(x) / (1 + exp(x))`.** Produces `NaN` for $x > 88.7$ in float32 and $x > 11.09$ in float16.
- **Passing a 0-D array directly.** Compilation fails on the first call with an error inside MLX's headers (`subscripted value is not an array`).
- **Using built-in MLX arithmetic.** Returning `nn.silu(a) * b` produces the right numbers, but the judge inspects the graph and rejects `Sigmoid` and `Multiply` nodes.
- **Exceeding the threadgroup limit.** Setting `threadgroup=(4096, 1, 1)` raises a runtime error at dispatch because Apple GPUs allow at most 1024 threads per threadgroup.

## Complexity

The kernel launches one thread per element and performs one read from each input and one write to the output, using no extra temporary buffers for contiguous inputs. With `ensure_row_contiguous=True`, each strided input adds one contiguous copy of itself.

## In the wild

`mx.fast.metal_kernel` builds a Metal library from the source string and caches it by name and source. Rebuilding an identical kernel takes about 0.2 ms compared to 44 ms for the initial compilation. If you interpolate dynamic values like tensor shapes into the source string, MLX recompiles on every call. Pass compile-time constants through `template` or read dynamic dimensions from `a_shape` instead. MLX's documentation builds a `grid_sample` kernel using this same API and reports an 8x speedup over composed built-in operations on an M1 Max, plus a 40x speedup for its backward pass using `atomic_outputs=True` and `init_value=0`.
