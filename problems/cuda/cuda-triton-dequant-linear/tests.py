"""Tests for cuda-triton-dequant-linear.

The inputs are made here, in NumPy: random 4-bit codes packed eight to a uint32 word,
lowest bits first (the layout of `bit-packing`, of `mx.quantize`, and of the Metal
kernel problem), and random scales and biases, one pair per group of columns. The
expected product is `x @ (codes * scale + bias).T` in float64 from exactly those
values, so the oracle is neither the reference kernel nor torch.

On the CPU the kernel runs under the kit's `tritonshim` (NumPy, one program after
another). The judge launches `qlinear_kernel` itself on a grid of `(n_out, batch)` with
block sizes of its choosing and checks:

- **the values** (the output starts as NaN, so an element never written shows);
- **the counters of the launch**: `n_out * batch` programs, exactly one element stored
  per program (the kernel has nowhere to put a dequantized weight, and may not
  overwrite its inputs: they are passed read-only), and at most `4 * n_in` elements
  loaded per program;
- what the shim refuses on its own: an unmasked access outside a tensor
  (`tl.OutOfBounds`), a masked-off value used without `other=` (NaN), two programs
  storing to one element (`tl.DataRace`).

Group sizes that do not divide the block, blocks longer than the row, codes with the
top bit set, and float16 inputs with a float32 output are where the mutants fail.

`qlinear`, the wrapper, is checked for shape, dtype, values, strided inputs, and for
one launch of the kernel on the right grid (`qlinear_kernel.last`).

The GPU cases (`requires="cuda"`) call `qlinear` on CUDA tensors, which makes the shim
hand the kernel's source to the real Triton compiler. Neither that path nor the shim's
agreement with Triton has ever been tested on a GPU.
"""

import numpy as np
import torch

from whetstone.judge import case, check
from whetstone.kit import tritonshim as tl

KERNEL = "qlinear_kernel"
FN = "qlinear"
TOL32 = {"rtol": 1e-4, "atol": 1e-4}
TOL16 = {"rtol": 2e-3, "atol": 2e-3}


def pack(codes):
    """(n_out, n_in) codes in 0..15 -> (n_out, n_in / 8) uint32 words, lowest bits first."""
    n_out, n_in = codes.shape
    grouped = codes.astype(np.uint64).reshape(n_out, n_in // 8, 8)
    shifts = np.arange(8, dtype=np.uint64) * np.uint64(4)
    return (grouped << shifts).sum(axis=-1).astype(np.uint32)


def expected(x, codes, scales, biases, group_size, absolute=False):
    """x @ W_hat.T in float64, from the values as given (or x @ |W_hat|.T)."""
    s = np.repeat(scales.astype(np.float64), group_size, axis=1)
    b = np.repeat(biases.astype(np.float64), group_size, axis=1)
    weights = codes.astype(np.float64) * s + b
    return x.astype(np.float64) @ (np.abs(weights) if absolute else weights).T


def layer(rng, n_out, n_in, group_size, dtype=np.float32, codes=None):
    """A random quantized layer: (codes, words, scales, biases)."""
    if codes is None:
        codes = rng.integers(0, 16, (n_out, n_in))
    n_groups = n_in // group_size
    # Weights of order 0.1: 16 levels of a scale of about 0.02, from a bias near -0.15.
    scales = rng.uniform(0.01, 0.03, (n_out, n_groups)) * rng.choice([-1.0, 1.0], (n_out, n_groups))
    biases = rng.uniform(-0.3, 0.3, (n_out, n_groups))
    return codes, pack(codes), scales.astype(dtype), biases.astype(dtype)


def the_kernel(ctx):
    kernel = getattr(ctx.solution, KERNEL)
    check.that(
        isinstance(kernel, tl.JITFunction),
        f"{KERNEL}: is not a kernel; decorate the function with @tl.jit",
    )
    return kernel


def check_counters(info, batch, n_out, n_in, label):
    programs = batch * n_out
    detail = (
        f"batch {batch}, n_out {n_out}, n_in {n_in}\n"
        f"programs: {info.programs}, elements loaded: {info.elements_loaded}, "
        f"elements stored: {info.elements_stored}"
    )
    check.that(
        info.programs == programs,
        f"{label}: the launch ran {info.programs} programs, which is not one per output element",
        detail,
    )
    check.that(
        info.elements_stored == programs,
        f"{label}: {info.elements_stored} elements were stored, which is not one per program",
        detail + "\nthe only memory the kernel writes is its own element of y",
    )
    check.that(
        info.elements_loaded <= 4 * n_in * programs,
        f"{label}: {info.elements_loaded} elements were loaded, more than 4 * n_in per program",
        detail + "\nper column: one activation, one word, one scale, one bias",
    )


def launch(ctx, x, codes, words, scales, biases, group_size, block, label=KERNEL, y_dtype=None):
    """Launch the solution's kernel on (n_out, batch) programs; check values and counters."""
    kernel = the_kernel(ctx)
    batch, n_in = x.shape
    n_out = words.shape[0]
    inputs = [np.ascontiguousarray(a).copy() for a in (x, words, scales, biases)]
    for array in inputs:
        array.setflags(write=False)
    y = np.full((batch, n_out), np.nan, dtype=np.float32 if y_dtype is None else y_dtype)
    info = tl.launch(kernel, (n_out, batch), *inputs, y, n_in, n_out, group_size, BLOCK=block)
    want = expected(x, codes, scales, biases, group_size)
    tolerance = dict(TOL16 if y.dtype == np.float16 else TOL32)
    # Sums of large products that cancel: the absolute tolerance follows their size.
    size = expected(np.abs(x), codes, scales, biases, group_size, absolute=True).max()
    tolerance["atol"] = max(tolerance["atol"], 2e-6 * float(size))
    check.allclose(y, want.astype(y.dtype), **tolerance, label=label)
    check_counters(info, batch, n_out, n_in, label)
    return y


def call(ctx, x, words, scales, biases, group_size, want, block=None, label=FN):
    """Call the wrapper on CPU tensors; check type, shape, dtype, values, and the launch."""
    kernel = the_kernel(ctx)
    kernel.last = None
    x_given = x.clone()
    extra = {} if block is None else {"block": block}
    out = ctx.solution.qlinear(x, words, scales, biases, group_size, **extra)
    check.that(isinstance(out, torch.Tensor), f"{label}: returned a {type(out).__name__}")
    check.that(
        tuple(out.shape) == want.shape,
        f"{label}: got shape {tuple(out.shape)}, which is not the expected shape",
        f"expected (..., n_out) = {want.shape}",
    )
    check.that(
        out.dtype == x.dtype,
        f"{label}: got dtype {str(out.dtype).removeprefix('torch.')}, which is not the "
        "expected dtype",
        f"expected the dtype of x, {str(x.dtype).removeprefix('torch.')}",
    )
    tolerance = TOL16 if x.dtype == torch.float16 else TOL32
    check.allclose(out.numpy(), want.astype(out.numpy().dtype), **tolerance, label=label)
    check.that(bool(torch.equal(x, x_given)), f"{label}: x was modified in place")
    info = kernel.last
    check.that(
        info is not None,
        f"{label}: {KERNEL} was not launched; the product must come from the kernel",
    )
    n_out, n_in = want.shape[-1], x.shape[-1]
    check_counters(info, x.numel() // n_in, n_out, n_in, label)
    return out


def sweep(ctx, batch, n_out, n_in, group_size, block, dtype=np.float32, rng=None, public=False):
    rng = ctx.rng if rng is None else rng
    codes, words, scales, biases = layer(rng, n_out, n_in, group_size, dtype)
    x = rng.standard_normal((batch, n_in)).astype(dtype)
    label = KERNEL
    if public:
        label = f"{KERNEL}(batch {batch}, {n_out} x {n_in}, group_size={group_size}, BLOCK={block})"
    return launch(ctx, x, codes, words, scales, biases, group_size, block, label=label)


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------

EXAMPLE_CODES = np.array([[5, 8, 0, 15, 0, 3, 15, 10], [0, 0, 0, 0, 15, 0, 7, 11]])
EXAMPLE_SCALES = np.array([[0.1, 0.06], [1.0, 0.05]], dtype=np.float32)
EXAMPLE_BIASES = np.array([[-0.5, 2.0], [-1.0, -0.25]], dtype=np.float32)
EXAMPLE_X = np.array(
    [[1.0, 2.0, -1.0, 0.5, 0.0, 1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
    dtype=np.float32,
)


@case("the worked example: a 2 x 8 layer in groups of 4, two batch rows", public=True)
def worked_example(ctx):
    words = pack(EXAMPLE_CODES)
    check.equal(words[:, 0].tolist(), [0xAF30F085, 0xB70F0000], label="authoring: packed words")
    y = launch(ctx, EXAMPLE_X, EXAMPLE_CODES, words, EXAMPLE_SCALES, EXAMPLE_BIASES, 4, 16)
    check.allclose(y, [[4.08, -2.95], [2.0, 0.5]], rtol=1e-5, atol=1e-5, label=KERNEL)


@case("a 12 x 96 layer in groups of 32, three batch rows, blocks of 64 and of 16", public=True)
def one_layer(ctx):
    # Rejects high_nibble_first, scales_of_row_zero, y_transposed, bias_outside_product.
    sweep(ctx, 3, 12, 96, 32, 64, public=True)
    sweep(ctx, 3, 12, 96, 32, 16, public=True)


@case("groups of 24 columns under blocks of 16 and 64: a block spans several groups",
      public=True)  # fmt: skip
def groups_cross_blocks(ctx):
    # Rejects group_from_block.
    sweep(ctx, 2, 5, 72, 24, 16, public=True)
    sweep(ctx, 2, 5, 72, 24, 64, public=True)
    sweep(ctx, 2, 4, 64, 8, 32, public=True)


@case("qlinear: a vector, a batch, and a (batch, sequence, n_in) tensor", public=True)
def wrapper_shapes(ctx):
    # Rejects materializes_weights (no launch) and grid_axes_swapped.
    rng = ctx.rng
    codes, words, scales, biases = layer(rng, 6, 80, 16)
    tensors = [torch.from_numpy(a) for a in (words, scales, biases)]
    for shape in ((80,), (4, 80), (2, 3, 80)):
        x = rng.standard_normal(shape).astype(np.float32)
        want = expected(x, codes, scales, biases, 16)
        call(ctx, torch.from_numpy(x), *tensors, 16, want, label=f"{FN}(x of shape {shape})")
    x = rng.standard_normal((5, 80)).astype(np.float32)
    call(ctx, torch.from_numpy(x), *tensors, 16, expected(x, codes, scales, biases, 16),
         block=32, label=f"{FN}(x of shape (5, 80), block=32)")  # fmt: skip


# Hidden cases.


@case("the largest and smallest codes: all fifteen, all zero, only the top bit, alternating")
def extreme_codes(ctx):
    # Rejects signed_shift: codes of 8 and above have their top bit set.
    rng = ctx.rng
    codes = np.empty((5, 64), dtype=np.int64)
    codes[0] = 15
    codes[1] = 0
    codes[2] = 8
    codes[3] = np.where(np.arange(64) % 2 == 0, 15, 0)
    codes[4] = np.arange(64) % 16
    _, words, scales, biases = layer(rng, 5, 64, 16, codes=codes)
    x = rng.standard_normal((2, 64)).astype(np.float32)
    launch(ctx, x, codes, words, scales, biases, 16, 32)
    one_hot = np.eye(64, dtype=np.float32)[[0, 7, 8, 63]]  # reads single weights
    launch(ctx, one_hot, codes, words, scales, biases, 16, 64)


@case("float16 activations, scales, and biases with a float32 output")
def half_precision_inputs(ctx):
    # Rejects float16_accumulator: the products are of order 100 and partly cancel.
    rng = ctx.rng
    codes, words, scales, biases = layer(rng, 6, 128, 32, dtype=np.float16)
    x = (rng.standard_normal((3, 128)) * 40.0).astype(np.float16)
    launch(ctx, x, codes, words, scales, biases, 32, 64, y_dtype=np.float32)
    launch(ctx, x * np.float16(20.0), codes, words, scales, biases, 32, 32, y_dtype=np.float32)
    sweep(ctx, 2, 4, 48, 16, 16, dtype=np.float16)


@case("blocks longer than the row, one output row, one batch row, blocks of one column")
def short_rows(ctx):
    # Rejects no_tail_mask and missing_other.
    for batch, n_out, n_in, group_size, block in (
        (1, 1, 8, 8, 64),
        (2, 3, 24, 8, 128),
        (1, 7, 40, 40, 16),
        (4, 1, 16, 2, 1),
        (1, 2, 8, 1, 4),
    ):
        sweep(ctx, batch, n_out, n_in, group_size, block)


@case("qlinear on strided inputs, and with float16 activations")
def wrapper_strided_and_half(ctx):
    # Rejects wrapper_not_contiguous and wrapper_float32_output.
    rng = ctx.rng
    codes, words, scales, biases = layer(rng, 8, 64, 16)
    x = rng.standard_normal((6, 64)).astype(np.float32)
    every_other = slice(None, None, 2)
    call(
        ctx,
        torch.from_numpy(x)[every_other],
        torch.from_numpy(words)[every_other],
        torch.from_numpy(scales)[every_other],
        torch.from_numpy(biases)[every_other],
        16,
        expected(x[::2], codes[::2], scales[::2], biases[::2], 16),
        block=32,
    )
    wide = torch.from_numpy(np.repeat(x, 2, axis=1))
    call(ctx, wide[:, ::2], *(torch.from_numpy(a) for a in (words, scales, biases)), 16,
         expected(x, codes, scales, biases, 16))  # fmt: skip
    codes, words, scales, biases = layer(rng, 4, 48, 24, dtype=np.float16)
    x = rng.standard_normal((2, 3, 48)).astype(np.float16)
    call(ctx, torch.from_numpy(x), *(torch.from_numpy(a) for a in (words, scales, biases)), 24,
         expected(x, codes, scales, biases, 24), block=16)  # fmt: skip


@case("random layers: group sizes, block sizes, and ragged tails")
def random_layers(ctx):
    groups = [1, 4, 8, 12, 24, 32]
    blocks = [2, 8, 32, 128]

    def prop(rng):
        group_size = groups[int(rng.integers(0, len(groups)))]
        unit = int(np.lcm(group_size, 8))
        n_in = unit * int(rng.integers(1, max(2, 120 // unit) + 1))
        block = blocks[int(rng.integers(0, len(blocks)))]
        dtype = np.float16 if rng.random() < 0.25 else np.float32
        sweep(ctx, int(rng.integers(1, 4)), int(rng.integers(1, 6)), n_in, group_size, block,
              dtype=dtype, rng=rng)  # fmt: skip

    check.forall(prop, trials=12)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine: the kernel goes to the real Triton.
# --------------------------------------------------------------------------------------


def on_gpu(ctx, batch, n_out, n_in, group_size, block, dtype):
    device = torch.device("cuda")
    rng = ctx.rng
    codes, words, scales, biases = layer(rng, n_out, n_in, group_size, dtype)
    x = rng.standard_normal((batch, n_in)).astype(dtype)
    out = ctx.solution.qlinear(
        torch.from_numpy(x).to(device),
        torch.from_numpy(words).to(device),
        torch.from_numpy(scales).to(device),
        torch.from_numpy(biases).to(device),
        group_size,
        block=block,
    )
    check.that(
        isinstance(out, torch.Tensor) and out.device.type == "cuda",
        f"{FN}: the result is not a tensor on the CUDA device",
    )
    check.that(
        tuple(out.shape) == (batch, n_out) and out.dtype == torch.from_numpy(x).dtype,
        f"{FN}: got shape {tuple(out.shape)} and dtype {out.dtype}, which is not what is expected",
    )
    want = expected(x, codes, scales, biases, group_size)
    tolerance = TOL16 if dtype == np.float16 else {"rtol": 1e-3, "atol": 1e-3}
    check.allclose(out.cpu().numpy(), want.astype(dtype), **tolerance, label=FN)


@case("on the GPU: a float32 layer of 12 x 96 in groups of 32, compiled by Triton",
      requires="cuda", time_limit_s=120)  # fmt: skip
def gpu_float32(ctx):
    on_gpu(ctx, 3, 12, 96, 32, 64, np.float32)
    on_gpu(ctx, 2, 5, 72, 24, 16, np.float32)


@case("on the GPU: float16 activations, scales, and biases", requires="cuda", time_limit_s=120)
def gpu_float16(ctx):
    on_gpu(ctx, 2, 8, 128, 32, 64, np.float16)
