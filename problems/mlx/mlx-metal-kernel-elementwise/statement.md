Write your first GPU kernel for MLX: one Metal function that computes
`silu(a) * b` for every element of two arrays, launched with one thread per element
by `mx.fast.metal_kernel`. The arithmetic is one line. The work is everything around
it: how many threads to launch, what each thread indexes, which type the kernel is
compiled for, and what MLX hands a kernel when the input is a strided view or has no
dimensions at all.

## Specification

```python
def fused_silu_mul(a, b): ...
```

For every index $i$ of `a`,

$$
\text{out}_i = \operatorname{silu}(a_i) \cdot b_i ,
\qquad
\operatorname{silu}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} .
$$

- `a`, `b`: two arrays of the same shape and the same dtype, **float16 or float32**,
  with zero to four dimensions. Either may have no elements. Either may be a strided
  view of a larger buffer: transposed, sliced with a step, reversed, or broadcast.
- Returns an array of the same shape and the **same dtype** as the inputs.

**The rule of the exercise.** The result is the output of one kernel built with
`mx.fast.metal_kernel`. The judge reads the lazy graph behind the returned array
before evaluating it: with evaluated inputs it must contain exactly one `CustomKernel`
operation and, besides it, only operations that rearrange data (reshape, flatten,
squeeze, expand_dims, transpose, slice, broadcast, `mx.contiguous`). MLX arithmetic
(`nn.silu`, `mx.sigmoid`, `*`) and dtype casts are not allowed, and the function must
not evaluate anything itself: it returns a lazy array, like every MLX operation.

**What `mx.fast.metal_kernel` does.** You pass the *body* of a Metal function as a
string and name its inputs and outputs; MLX writes the signature. Calling the
returned object launches the kernel:

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

Three facts about the generated function, all of MLX 0.32.4:

- The pointer type of each input is the input's dtype (`half` for float16, `float`
  for float32) and that of each output is its entry in `output_dtypes`. A template
  parameter such as `T` is a name the body can use for its own variables.
- An input with **zero dimensions** is passed by value (`const float& a`), not as a
  buffer: `a[elem]` does not compile for it.
- With `ensure_row_contiguous=True`, element `i` of every input, in row-major order,
  is at `a[i]`. With `ensure_row_contiguous=False` no copy is made, the kernel
  receives the raw buffer of a view, and the body must compute where element `i`
  lives from `a_shape`, `a_strides`, and `a_ndim`, which MLX adds to the signature
  when the body mentions them. Either setting is accepted here.

Whatever the kernel does not write is not initialized: an output element that no
thread reaches holds whatever was in that memory before.

## Shapes

| name | shape | dtype |
|---|---|---|
| `a` | `(d_1, ..., d_k)`, $0 \le k \le 4$, any $d_j \ge 0$ | float16 or float32 |
| `b` | the shape of `a` | the dtype of `a` |
| result | the shape of `a` | the dtype of `a` |

## Example

```python
a = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
b = mx.array([1.0, 2.0, 3.0, 4.0, 0.5, -1.0])

fused_silu_mul(a, b)
# array([-0.238406, -0.537883, 0, 2.92423, 0.880797, -2.85772], dtype=float32)
```

Six threads run, one per element. Thread 3 reads $a_3 = 1$ and $b_3 = 4$:
$\sigma(1) = 0.731059$, so it writes $1 \cdot 0.731059 \cdot 4 = 2.92423$. Thread 0
reads $-2$ and $1$: $-2 \cdot \sigma(-2) = -2 \cdot 0.119203 = -0.238406$.

For a `(64, 48)` input the kernel needs $64 \cdot 48 = 3072$ threads, and for
`a.T` of that matrix, element 1 of the view (row 0, column 1 of the transpose) is
element 48 of the buffer underneath.

## Constraints

- At most $2^{21}$ elements. Values of `a` lie in $[-10^4, 10^4]$ for float32 and in
  $[-20, 20]$ for float16; values of `b` lie in $[-4, 4]$.
- The result is compared with $a_i / (1 + e^{-a_i}) \cdot b_i$ computed in float64
  from the values the function received: to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6` for float32, and to a relative tolerance of `5e-3` and
  an absolute tolerance of `2e-3` for float16 (arithmetic in half precision is good
  to about three digits). A NaN is always wrong: $e^{x}$ overflows float16 above
  $x = 11.09$ and float32 above $x = 88.7$, inside the range of the inputs.
- The graph rule above is checked on every call whose inputs are evaluated.
- Every case needs the Apple GPU: a Metal kernel has no CPU implementation.

## Why this matters

MLX's built-in operations cover almost everything an LLM needs, and the cases they do
not cover are exactly where a runtime differentiates itself: a dequantizer for a
format MLX does not ship, a sampler that filters logits in one pass, a fused
activation in a hot loop, a custom attention variant. On Apple silicon the way to add
such an operation without leaving Python is `mx.fast.metal_kernel`, and `mlx-lm` and
MLX's own `fast` module are built from kernels of this shape. The mistakes are the
ones every GPU programmer makes once: a grid that covers the first dimension and not
the array, a kernel compiled for `float` that silently doubles the memory of a
half-precision model, an index that is right until someone passes a transposed view.
None of them raises an error. Each returns an array of the right shape with wrong or
stale numbers in it.
