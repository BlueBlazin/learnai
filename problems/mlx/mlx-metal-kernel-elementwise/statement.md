Write a custom GPU kernel for MLX using `mx.fast.metal_kernel` that computes `silu(a) * b` elementwise, launching one thread per element. Your implementation must configure the launch grid, thread indexing, and dtype templating, and handle strided views and 0-D scalar inputs across float16 and float32 arrays.

## Specification

```python
def fused_silu_mul(a, b): ...
```

For every element index $i$ of `a` and `b`, compute

$$
\text{out}_i = \operatorname{silu}(a_i) \cdot b_i ,
\qquad
\operatorname{silu}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} .
$$

`a` and `b` have the same shape and the same dtype (`float16` or `float32`), with zero to four dimensions. Either array may be empty (zero elements) or a strided view of a larger buffer, such as a transposed, stepped-slice, reversed, or broadcast view. Return an array with the same shape and dtype as the inputs.

**Kernel requirement.** The result must come from a single kernel built with `mx.fast.metal_kernel`. Before evaluating the returned array, the judge inspects its lazy graph on evaluated inputs. That graph must contain exactly one `CustomKernel` operation and, aside from it, only view or layout operations (reshape, flatten, squeeze, expand_dims, transpose, slice, broadcast, or `mx.contiguous`). Built-in MLX arithmetic (`nn.silu`, `mx.sigmoid`, `*`) and dtype casts are forbidden. Do not evaluate anything inside `fused_silu_mul`, because it must return a lazy array like any other MLX operation.

**How `mx.fast.metal_kernel` works.** You supply the *body* of a Metal function as a string along with its input and output names, and MLX generates the function signature. Calling the returned object dispatches the kernel:

```python
kernel = mx.fast.metal_kernel(
    name="...",
    input_names=["a", "b"],  # each becomes `const device <type>* a`
    output_names=["out"],  # each becomes `device <type>* out`
    source="...",  # the body; may use thread_position_in_grid
    ensure_row_contiguous=True,  # the default: strided inputs are copied first
)
(out,) = kernel(
    inputs=[a, b],
    template=[("T", mx.float32)],  # template parameters the body may use
    grid=(n, 1, 1),  # launch n threads; thread_position_in_grid.x runs over 0..n-1
    threadgroup=(256, 1, 1),  # threads per group; at most 1024 in total
    output_shapes=[shape],
    output_dtypes=[dtype],
)
```

Keep three details of MLX 0.32.4 in mind when writing the kernel body:

- Each input pointer type matches the input array's dtype (`half` for float16, `float` for float32), and each output pointer type matches its entry in `output_dtypes`. Template parameters like `T` let the kernel body declare variables with the matching type.
- A **0-D** input is passed by value (`const float& a`) rather than as a buffer pointer, so `a[i]` fails to compile for 0-D arrays.
- With `ensure_row_contiguous=True` (the default), MLX copies strided inputs into row-major order first, so element `i` is always at `a[i]`. With `ensure_row_contiguous=False`, no copy is made and the kernel receives the raw underlying buffer, so the body must compute each offset using `a_shape`, `a_strides`, and `a_ndim` (which MLX adds to the signature whenever the body references them). Either setting is accepted.

Output buffers are uninitialized. Any output element that no thread writes retains whatever data was previously in that memory.

## Shapes

| name | shape | dtype |
|---|---|---|
| `a` | `(d_1, ..., d_k)`, $0 \le k \le 4$, each $d_j \ge 0$ | float16 or float32 |
| `b` | same as `a` | same as `a` |
| result | same as `a` | same as `a` |

## Example

```python
a = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
b = mx.array([1.0, 2.0, 3.0, 4.0, 0.5, -1.0])

fused_silu_mul(a, b)
# array([-0.238406, -0.537883, 0, 2.92423, 0.880797, -2.85772], dtype=float32)
```

Six threads run, one per element. Thread 3 reads $a_3 = 1$ and $b_3 = 4$, computes $\sigma(1) = 0.731059$, and writes $1 \cdot 0.731059 \cdot 4 = 2.92423$. Thread 0 reads $-2$ and $1$ and writes $-2 \cdot \sigma(-2) = -2 \cdot 0.119203 = -0.238406$.

For a `(64, 48)` input, the kernel needs $64 \cdot 48 = 3072$ threads. If you pass `a.T` for that matrix, element 1 of the transposed view (row 0, column 1) sits at element 48 of the underlying buffer.

## Constraints

- At most $2^{21}$ elements. Values of `a` lie in $[-10^4, 10^4]$ for float32 and $[-20, 20]$ for float16. Values of `b` lie in $[-4, 4]$.
- Outputs are compared against $a_i / (1 + e^{-a_i}) \cdot b_i$ computed in float64 from the input values. Float32 must match within relative tolerance `1e-4` and absolute tolerance `1e-6`. Float16 must match within relative tolerance `5e-3` and absolute tolerance `2e-3` (half precision carries about three decimal digits).
- NaNs are always rejected. Note that $e^x$ overflows float16 above $x = 11.09$ and float32 above $x = 88.7$, both of which lie inside the input range.
- The graph rule above is checked on every call whose inputs are evaluated.
- Every test case requires an Apple GPU, because Metal kernels have no CPU implementation.

## Why this matters

MLX's built-in operations cover most standard layers, but custom runtimes often need operations MLX does not ship, such as a dequantizer for a custom format, a single-pass logit sampler, a fused activation in a hot loop, or a custom attention variant. On Apple silicon, `mx.fast.metal_kernel` lets you add those operations without leaving Python, and both `mlx-lm` and MLX's own `fast` module rely on it.

The bugs in this exercise are the ones every GPU programmer hits at least once. A grid sized to the first axis leaves the rest of a matrix uninitialized, a hardcoded `float` template silently doubles memory usage in a half-precision model, and raw linear indexing breaks as soon as a caller passes a transposed view. None of these raises an exception. Each returns an array of the expected shape filled with stale or corrupted numbers.
