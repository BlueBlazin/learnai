"""Tests for cuda-triton-rmsnorm.

On the CPU the kernel runs under the kit's `tritonshim`: the same kernel source,
executed in NumPy one program after another. The shim is strict where a GPU is silent:
an unmasked access outside the tensor raises `tl.OutOfBounds`, a masked-off element of
a load without `other=` is NaN, two programs storing to the same element raise
`tl.DataRace`, and a store through a read-only tensor raises.

The judge launches `rmsnorm_kernel` itself, on a grid of `(n_rows,)` and with block
sizes of its choosing, and checks:

- **the values**, against the formula computed in float64 from the values the kernel
  was given (so the oracle is neither the reference kernel nor torch). The output
  starts as NaN, so an element that is never written shows;
- **the counters of the launch**: one program per row, every output element stored
  exactly once, at most three element loads per output element (x twice, the gain
  once);
- that `x` and the gains are only read (they are passed read-only).

Row lengths that are not a multiple of the block, rows shorter than one block, and
the last row of the tensor are where a missing or wrong tail mask shows. Small
activations show the place of eps; float16 activations above 255 show an accumulator
in the wrong type.

`rms_norm`, the wrapper, is checked for shape, dtype, values, strided inputs, and for
having launched the kernel exactly once with one program per row
(`rmsnorm_kernel.last`).

The GPU cases (`requires="cuda"`) call `rms_norm` on CUDA tensors, which makes the shim
hand the kernel's source to the real Triton compiler. Neither that path nor the shim's
agreement with Triton has ever been tested on a GPU.
"""

import numpy as np
import torch

from whetstone.judge import case, check
from whetstone.kit import tritonshim as tl

KERNEL = "rmsnorm_kernel"
FN = "rms_norm"
TOL = {
    np.dtype(np.float32): {"rtol": 1e-4, "atol": 1e-6},
    np.dtype(np.float16): {"rtol": 2e-3, "atol": 1e-3},
}


def expected(x, w, eps):
    """The formula in float64, from the values as given; rounded once to x's dtype."""
    x64, w64 = x.astype(np.float64), w.astype(np.float64)
    out = x64 / np.sqrt((x64 * x64).mean(axis=-1, keepdims=True) + eps) * w64
    return out.astype(x.dtype)


def the_kernel(ctx):
    kernel = getattr(ctx.solution, KERNEL)
    check.that(
        isinstance(kernel, tl.JITFunction),
        f"{KERNEL}: is not a kernel; decorate the function with @tl.jit",
    )
    return kernel


def launch(ctx, x, w, eps, block, label=KERNEL):
    """Launch the solution's kernel on (n_rows,) programs; check values and counters."""
    kernel = the_kernel(ctx)
    n_rows, n_cols = x.shape
    x, w = np.ascontiguousarray(x), np.ascontiguousarray(w)
    x_given, w_given = x.copy(), w.copy()
    x.setflags(write=False)
    w.setflags(write=False)
    out = np.full_like(x, np.nan)
    info = tl.launch(kernel, (n_rows,), x, w, out, n_cols, eps, BLOCK=block)
    check.allclose(out, expected(x_given, w_given, eps), **TOL[x.dtype], label=label)
    check_counters(info, n_rows, n_cols, label)
    return out


def check_counters(info, n_rows, n_cols, label):
    n = n_rows * n_cols
    detail = (
        f"{n_rows} rows of {n_cols} columns\n"
        f"programs: {info.programs}, elements loaded: {info.elements_loaded}, "
        f"elements stored: {info.elements_stored}"
    )
    check.that(
        info.programs == n_rows,
        f"{label}: the launch ran {info.programs} programs, which is not one per row",
        detail,
    )
    check.that(
        info.elements_stored == n,
        f"{label}: {info.elements_stored} elements were stored, which is not every output "
        "element exactly once",
        detail,
    )
    check.that(
        info.elements_loaded <= 3 * n,
        f"{label}: {info.elements_loaded} elements were loaded, more than 3 per output element",
        detail + "\nthe limit is 3 * n_rows * n_cols: x in each of two passes, the gain once",
    )


def call(ctx, x, w, eps=None, block=None, label=FN, want=None):
    """Call the wrapper on CPU tensors; check type, shape, dtype, values, and the launch."""
    kernel = the_kernel(ctx)
    kernel.last = None
    x_given = x.clone()
    extra = {}
    if eps is not None:
        extra["eps"] = eps
    if block is not None:
        extra["block"] = block
    out = ctx.solution.rms_norm(x, w, **extra)
    check.that(isinstance(out, torch.Tensor), f"{label}: returned a {type(out).__name__}")
    check.that(
        tuple(out.shape) == tuple(x.shape),
        f"{label}: got shape {tuple(out.shape)}, which is not the expected shape",
        f"expected the shape of x, {tuple(x.shape)}",
    )
    check.that(
        out.dtype == x.dtype,
        f"{label}: got dtype {str(out.dtype).removeprefix('torch.')}, which is not the "
        "expected dtype",
        f"expected the dtype of x, {str(x.dtype).removeprefix('torch.')}",
    )
    if want is None:
        want = expected(x_given.numpy(), w.numpy(), 1e-5 if eps is None else eps)
    check.allclose(out.numpy(), want, **TOL[np.dtype(want.dtype)], label=label)
    check.that(bool(torch.equal(x, x_given)), f"{label}: x was modified in place")
    info = kernel.last
    check.that(
        info is not None,
        f"{label}: {KERNEL} was not launched; the result must come from the kernel",
    )
    n_cols = x.shape[-1]
    check_counters(info, x.numel() // n_cols, n_cols, label)
    return out


def draw(rng, n_rows, n_cols, dtype=np.float32, scale=1.0):
    """Activations with rows of very different magnitude, and gains around 1."""
    row_scale = scale * 10.0 ** rng.uniform(-1.0, 1.0, (n_rows, 1))
    x = (rng.standard_normal((n_rows, n_cols)) * row_scale).astype(dtype)
    w = rng.uniform(0.5, 1.5, n_cols).astype(dtype)
    return x, w


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: two rows of five columns in blocks of four", public=True)
def worked_example(ctx):
    x = np.array([[3.0, 4.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    w = np.array([1.0, 0.5, 1.0, 1.0, 2.0], dtype=np.float32)
    out = launch(ctx, x, w, 1e-5, 4)
    check.allclose(
        out,
        [[1.341640, 0.894426, 0.0, 0.0, 0.0], [0.999995, 0.499998, 0.999995, 0.999995, 1.999990]],
        rtol=1e-5,
        label=KERNEL,
    )


@case("row lengths 5, 16, 37, and 100 in blocks of 16", public=True)
def ragged_rows(ctx):
    # Rejects other_one (37 and 100 are not multiples of 16), mean_over_padded_length,
    # mask_off_by_one, and single_block_only.
    for n_cols in (5, 16, 37, 100):
        x, w = draw(ctx.rng, 6, n_cols)
        launch(ctx, x, w, 1e-5, 16, label=f"{KERNEL}(6 rows of {n_cols}, BLOCK=16)")


@case("small activations: eps is inside the square root", public=True)
def small_activations(ctx):
    # Rejects eps_outside_root.
    x = np.array([[3e-3, 4e-3, 0.0], [1e-3, -2e-3, 2e-3], [0.0, 0.0, 0.0]], dtype=np.float32)
    w = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    launch(ctx, x, w, 1e-5, 4)
    x, w = draw(ctx.rng, 4, 24, scale=1e-3)
    launch(ctx, x, w, 1e-3, 8)


@case("rms_norm: a vector, a matrix, and a (batch, sequence, d) tensor", public=True)
def wrapper_shapes(ctx):
    # Rejects one_program_per_element (the launch's program count).
    rng = ctx.rng
    for shape in ((7,), (4, 37), (2, 3, 20)):
        x, w = draw(rng, int(np.prod(shape[:-1])), shape[-1])
        x = torch.from_numpy(x.reshape(shape))
        call(ctx, x, torch.from_numpy(w), label=f"{FN}(x of shape {shape})")
    x, w = draw(rng, 5, 70)
    call(ctx, torch.from_numpy(x), torch.from_numpy(w), eps=1e-3, block=16,
         label=f"{FN}(x of shape (5, 70), eps=1e-3, block=16)")  # fmt: skip


# Hidden cases.


@case("float16 activations up to 2000: the squares need float32")
def half_precision(ctx):
    # Rejects no_float32_cast: 2000 ** 2 overflows float16.
    rng = ctx.rng
    x = (rng.standard_normal((5, 37)) * 600.0).astype(np.float16)
    x[0, 3], x[4, 36] = 2000.0, -1900.0
    w = rng.uniform(0.5, 1.5, 37).astype(np.float16)
    launch(ctx, x, w, 1e-5, 16)
    x, w = draw(rng, 4, 20, dtype=np.float16)
    launch(ctx, x, w, 1e-5, 8)
    # Rejects wrapper_float32_output.
    x, w = draw(rng, 6, 12, dtype=np.float16)
    call(ctx, torch.from_numpy(x.reshape(2, 3, 12)), torch.from_numpy(w), block=8)


@case("blocks longer than the row, a single row, and a single column")
def short_rows(ctx):
    # Rejects no_tail_mask and store_unmasked: the last program's block passes the end
    # of the tensor, and every other program's block reaches into the next row.
    rng = ctx.rng
    for n_rows, n_cols, block in ((3, 5, 64), (1, 9, 16), (4, 1, 2), (2, 3, 1), (1, 64, 64)):
        x, w = draw(rng, n_rows, n_cols)
        launch(ctx, x, w, 1e-5, block)


@case("rows of very different magnitude, and rows of zeros")
def per_row_statistics(ctx):
    rng = ctx.rng
    x, w = draw(rng, 8, 40)
    x[0] *= 1e3
    x[3] = 0.0
    x[5] *= 1e-3
    x[7, :39] = 0.0  # the whole norm sits in the last column, in the tail block
    launch(ctx, x, w, 1e-6, 16)
    launch(ctx, x, w, 1e-2, 32)


@case("rms_norm on strided inputs: every second column, a transposed batch, a strided gain")
def strided_inputs(ctx):
    # Rejects wrapper_not_contiguous: the kernel indexes raw memory, not a view.
    rng = ctx.rng
    base, w = draw(rng, 6, 48)
    base, wide = torch.from_numpy(base), torch.from_numpy(np.repeat(w, 2))
    view = base[:, ::2]
    call(ctx, view, wide[: 24 * 2 : 2], want=expected(view.numpy(), w[:24], 1e-5))
    cube = torch.from_numpy(draw(rng, 12, 10)[0].reshape(3, 4, 10))
    gain = torch.from_numpy(w[:10].copy())
    turned = cube.transpose(0, 1)
    call(ctx, turned, gain, block=4, want=expected(turned.numpy(), w[:10], 1e-5))
    rows = base[::2]
    call(ctx, rows, wide[::2], block=32, want=expected(rows.numpy(), w, 1e-5))


@case("random shapes and block sizes")
def random_layouts(ctx):
    shapes = [(1, 1), (3, 7), (2, 33), (5, 64), (4, 130)]
    blocks = [1, 4, 32, 128]

    def prop(rng):
        n_rows, n_cols = shapes[int(rng.integers(0, len(shapes)))]
        block = blocks[int(rng.integers(0, len(blocks)))]
        dtype = np.float16 if rng.random() < 0.3 else np.float32
        x, w = draw(rng, n_rows, n_cols, dtype=dtype)
        launch(ctx, x, w, float(10.0 ** rng.uniform(-6, -3)), block)

    check.forall(prop, trials=12)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine: the kernel goes to the real Triton.
# --------------------------------------------------------------------------------------


def on_gpu(ctx, shape, dtype, block):
    device = torch.device("cuda")
    x, w = draw(ctx.rng, int(np.prod(shape[:-1])), shape[-1], dtype=dtype)
    x = x.reshape(shape)
    out = ctx.solution.rms_norm(torch.from_numpy(x).to(device), torch.from_numpy(w).to(device),
                                block=block)  # fmt: skip
    check.that(
        isinstance(out, torch.Tensor) and out.device.type == "cuda",
        f"{FN}: the result is not a tensor on the CUDA device",
    )
    check.that(
        tuple(out.shape) == shape and out.dtype == torch.from_numpy(x).dtype,
        f"{FN}: got shape {tuple(out.shape)} and dtype {out.dtype}, which is not what is expected",
    )
    tolerance = {"rtol": 1e-3, "atol": 1e-5} if dtype == np.float32 else TOL[np.dtype(dtype)]
    check.allclose(out.cpu().numpy(), expected(x, w, 1e-5), **tolerance, label=FN)


@case("on the GPU: float32 rows of 100 and of 37 columns, compiled by Triton", requires="cuda",
      time_limit_s=120)  # fmt: skip
def gpu_float32(ctx):
    on_gpu(ctx, (8, 100), np.float32, 64)
    on_gpu(ctx, (2, 3, 37), np.float32, 16)


@case("on the GPU: float16 activations", requires="cuda", time_limit_s=120)
def gpu_float16(ctx):
    on_gpu(ctx, (4, 70), np.float16, 32)
