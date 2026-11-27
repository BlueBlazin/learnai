A Metal kernel runs once per thread, and `thread_position_in_grid` is the only value that distinguishes one thread from another. If each thread computes one output element, how many threads does a `(64, 48)` array need, and what index does a thread use in a row-major buffer? Does the tensor's multi-dimensional shape matter at all?

---

Launch `a.size` threads in a 1-D grid with `grid=(a.size, 1, 1)`. Let each thread compute `elem = thread_position_in_grid.x`, read `a[elem]` and `b[elem]`, and write `out[elem]`. Leave `ensure_row_contiguous` at its default (`True`) so MLX copies strided views into row-major order before dispatch. Pass the input dtype both in `template=[("T", a.dtype)]` and in `output_dtypes=[a.dtype]`. Write `silu(x)` as `x / (1 + exp(-x))`, because writing the sigmoid as `exp(x) / (1 + exp(x))` turns into `inf / inf` once `exp(x)` overflows. Finally, reshape 0-D inputs to `(1,)` before calling the kernel and reshape the output back to `()`, since 0-D arrays cannot be subscripted inside a Metal kernel.

---

The kernel body and dispatch call:

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

Construct `kernel` once outside `fused_silu_mul`. When `a.ndim == 0`, run the kernel on `a.reshape(1)` and `b.reshape(1)`, then return `out.reshape(())`.
