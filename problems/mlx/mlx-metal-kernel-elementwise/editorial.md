## The idea

`mx.fast.metal_kernel` turns a string into a GPU function. You write the body; MLX
writes the signature from what you declared and from what the body mentions. For the
reference solution and float32 inputs, `verbose=True` on the call prints:

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

Everything else follows from reading that signature.

- **The grid is the loop.** Metal calls the function once per point of `grid`, and
  `thread_position_in_grid` is the loop variable. An elementwise kernel needs one
  thread per element, so the grid is `(a.size, 1, 1)` whatever the shape: the output
  is always a row-major buffer, and the flat element number is all a thread needs.
  `threadgroup` only says how Metal batches the threads (at most 1024 per group; 256
  is a fine default for a 1-D grid). It does not change what is computed.
- **The buffers are typed by the arrays, the body by the template.** `a` is
  `const device half*` when `a` is float16. `T` is a name for the body's own
  variables; with `template=[("T", a.dtype)]` the arithmetic runs in the precision of
  the data, and with `output_dtypes=[a.dtype]` the result has the size of the input.
- **Outputs are not initialized.** A thread that does not exist writes nothing, and
  that part of the output holds whatever the allocator handed out: zeros from a fresh
  buffer, or a previous result from a recycled one.
- **A 0-d array is not a buffer.** MLX passes it as `const float& a`. Reshape to one
  element on the way in and back to `()` on the way out; both are free.
- **The result is lazy.** The call returns at once with a `CustomKernel` node; Metal
  compiles the function the first time it runs, and the kernel fuses nothing with its
  neighbors. That is why it is worth building once and reusing.

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

The sigmoid is written as $x / (1 + e^{-x})$. For large positive $x$ the exponential
underflows to 0 and the result is $x$. For very negative $x$ it overflows to `inf`,
but in the denominator, so the quotient is $-0$, which is the correct limit. The
other textbook form, $e^{x} / (1 + e^{x})$, is `inf / inf = NaN` as soon as $e^x$
overflows, and in half precision that happens at $x = 11.09$, an ordinary activation.

## Strided inputs: copy, or do the arithmetic

`a.T` is not a new buffer. It is the old buffer with shape `(48, 64)` and strides
`(1, 48)`: element $(r, c)$ of the view is at offset $r \cdot 1 + c \cdot 48$.

With `ensure_row_contiguous=True` (the default) MLX copies such an input into
row-major order before the kernel runs, and `a[elem]` is right. With
`ensure_row_contiguous=False` the kernel gets the buffer as it is, and the body has
to convert the element number into an offset. MLX includes a helper for that and
adds the three extra arguments when the body names them:

```cpp
T x = a[elem_to_loc(elem, a_shape, a_strides, a_ndim)];
```

`elem_to_loc` peels `elem` apart with the shape, last axis first, and adds up
`index * stride`. It computes in unsigned integers, and that is a trap in MLX 0.32.4:
a reversed view (`a[::-1]`) has a **negative** stride, the sum wraps, and the kernel
reads from the wrong place. The version that is correct for every view does the same
loop in signed 64-bit arithmetic:

```cpp
int64_t loc = 0;
uint rest = elem;
for (int d = a_ndim - 1; d >= 0; --d) {
    loc += int64_t(rest % uint(a_shape[d])) * a_strides[d];
    rest /= uint(a_shape[d]);
}
T x = a[loc];
```

Which one to use is a real trade. Measured on this machine (M1 Max, MLX 0.32.4) on
two transposed `2048 x 2048` float32 inputs, 16 MiB each:

| kernel | peak memory above the inputs | time per call |
|---|---|---|
| default (MLX copies both views) | 48 MiB | 0.62 ms |
| strided (`ensure_row_contiguous=False`) | 16 MiB | 1.32 ms |
| contiguous inputs, either kernel | 16 MiB | 0.45 ms |

The copy is fast, because it is a simple kernel with sequential writes, and it costs
a full copy of every strided input. The strided kernel allocates only the output and
pays for scattered reads and a division loop per element. For a kernel that runs on
activations, take the default. Turn the copy off when an input is large and usually
strided, and then test it with reversed views.

For scale: the three MLX operations `mx.sigmoid(a) * a * b` take 0.47 ms on the same
contiguous inputs. Fusing an elementwise chain by hand buys little on a GPU that is
limited by memory bandwidth. Custom kernels pay when MLX has no operation for what
you need, or when the unfused version would materialize something large; the next
problem is one of those.

## Common mistakes

- **A grid of `a.shape[0]` threads** (or of `a.shape`, used as a 1-D index). It is
  right for vectors, so the first test passes. For a `(64, 48)` matrix 64 of 3072
  elements are written and the rest is whatever was in the buffer.
- **Compiling for `float` regardless of the input.** `template=[("T", mx.float32)]`
  with `output_dtypes=[mx.float32]` reads a float16 input correctly (the pointer type
  follows the array) and returns float32: the right numbers at twice the memory, and
  every later operation of a half-precision model is promoted with it.
- **Turning off `ensure_row_contiguous` and keeping `a[elem]`.** Contiguous inputs
  still work. A transposed input is read in buffer order, so the output is the
  transpose's values in the wrong places.
- **Trusting `elem_to_loc` for every view.** Correct for transposed, stepped, and
  broadcast inputs; wrong for a negative stride.
- **`exp(x) / (1 + exp(x))`.** NaN for $x > 88.7$ in float32 and $x > 11.09$ in
  float16.
- **Passing a 0-d array.** The kernel fails to compile at the first call, with an
  error that points into MLX's headers: "subscripted value is not an array".
- **Returning MLX arithmetic.** `nn.silu(a) * b` has the right values and is not a
  kernel; the judge sees `Sigmoid` and `Multiply` in the graph.
- **A threadgroup above the device limit.** `threadgroup=(4096, 1, 1)` raises at
  dispatch: at most 1024 threads per group on this GPU.

## Complexity

One thread and one read of each input per element; no temporaries. With the default
setting, a strided input costs one extra copy of itself.

## In the wild

`mx.fast.metal_kernel` constructs a Metal library from the source and MLX caches it
by name and source: building the same kernel again costs 0.2 ms here, against 44 ms
for the first compilation. A kernel whose name or source changes on every call (a
shape formatted into the string, say) recompiles every time; pass sizes as template
integers or read them from `a_shape`. MLX's documentation develops a `grid_sample`
kernel the same way and reports an 8x speedup over the composition of built-in
operations on an M1 Max, and a 40x speedup for its gradient, which accumulates with
`atomic_outputs=True` into outputs zeroed by `init_value=0`.
