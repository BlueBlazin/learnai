A kernel is a function that runs once per thread, and the only thing that
distinguishes one thread from another is `thread_position_in_grid`. If every thread
handles one element, how many threads does an array of shape `(64, 48)` need, and
which number does a thread need to find its element in a row-major buffer? Does the
shape matter at all?

---

Launch `a.size` threads in a one-dimensional grid, `grid=(a.size, 1, 1)`, and let
thread `elem = thread_position_in_grid.x` read `a[elem]` and `b[elem]` and write
`out[elem]`. Leave `ensure_row_contiguous` at its default so that a strided view is
copied into row-major order before the kernel runs. Pass the dtype of the inputs as
a template parameter (`template=[("T", a.dtype)]`) and as the output dtype, and
write the sigmoid as `x / (1 + exp(-x))`: the form `exp(x) / (1 + exp(x))` is
`inf / inf` once `exp(x)` overflows. A 0-d array cannot be indexed inside a kernel;
reshape it to one element first and reshape the result back.

---

The body and the call:

```python
source = """
    uint elem = thread_position_in_grid.x;
    T x = a[elem];
    out[elem] = x / (T(1) + metal::exp(-x)) * b[elem];
"""
kernel = mx.fast.metal_kernel(
    name="fused_silu_mul", input_names=["a", "b"], output_names=["out"], source=source
)

out = kernel(
    inputs=[a, b],
    template=[("T", a.dtype)],
    grid=(a.size, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[a.shape],
    output_dtypes=[a.dtype],
)[0]
```

Build the kernel once, outside the function. For `a.ndim == 0`, call it on
`a.reshape(1)` and `b.reshape(1)` and return `out.reshape(())`.
