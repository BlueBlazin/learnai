"""tritonshim: a numpy stand-in for the subset of triton.language the CUDA track uses.

The semantics tested here follow the Triton documentation (triton-lang.org); the shim
has not been compared with real Triton, which this machine cannot install.
"""

import sys
import types

import numpy as np
import pytest
import torch

from whetstone.kit import tritonshim as tl


@tl.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # The vector-add tutorial of the Triton documentation, verbatim but for the decorator.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * n_cols
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + offs, mask=offs < n_cols, other=0.0)
        acc += x * x
    rms = tl.sqrt(tl.sum(acc) / n_cols + eps)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + base + offs, x / rms * w, mask=mask)


@tl.jit
def rmsnorm_no_mask(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * n_cols + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x / tl.sqrt(tl.sum(x * x) / n_cols + eps))


@tl.jit
def unpack_kernel(w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # 4-bit codes packed eight to a uint32 word, low bits first.
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n
    word = tl.load(w_ptr + cols // 8, mask=mask, other=0)
    code = (word >> ((cols % 8) * 4).to(tl.uint32)) & 0xF
    tl.store(out_ptr + cols, code.to(tl.float32), mask=mask)


@tl.jit
def probe(out_ptr, a_ptr, n, flag, eps, K: tl.constexpr):
    """Writes facts about types and values for the semantics tests."""
    i = tl.arange(0, 8) - 4
    tl.store(out_ptr + tl.arange(0, 8), i // 3)
    tl.store(out_ptr + 8 + tl.arange(0, 8), i % 3)
    a = tl.load(a_ptr + tl.arange(0, 4))  # uint8
    tl.store(out_ptr + 16, tl.sum(a))  # widened to uint32: 4 * 200 = 800
    tl.store(out_ptr + 17, tl.max(a + 100))  # stays uint8: wraps
    tl.store(out_ptr + 18, tl.sum(tl.where(a > 100, 1, 0)))
    tl.store(out_ptr + 19, K // 3)  # constexpr: Python semantics
    tl.store(out_ptr + 20, tl.num_programs(0) * 10 + tl.program_id(1))
    if flag:
        tl.store(out_ptr + 21, 7)


def launch_probe(**kwargs):
    out = np.zeros(32, dtype=np.int32)
    a = np.full(4, 200, dtype=np.uint8)
    tl.launch(probe, (1,), out, a, 4, True, 1e-5, **kwargs)
    return out


def test_the_vector_add_tutorial_runs_and_writes_in_place():
    rng = np.random.default_rng(0)
    x, y = (torch.tensor(rng.standard_normal(100).astype(np.float32)) for _ in range(2))
    out = torch.zeros(100)
    info = tl.launch(add_kernel, (tl.cdiv(100, 32),), x, y, out, 100, BLOCK_SIZE=32)
    assert torch.equal(out, x + y)
    assert info.snapshot() == {
        "programs": 4, "loads": 8, "stores": 4, "elements_loaded": 200, "elements_stored": 100,
    }  # fmt: skip
    # Triton's own spelling, with a grid computed from the meta-parameters.
    out.zero_()
    add_kernel[lambda meta: (tl.cdiv(100, meta["BLOCK_SIZE"]),)](x, y, out, 100, BLOCK_SIZE=64)
    assert torch.equal(out, x + y)
    assert add_kernel.last.programs == 2


@pytest.mark.parametrize("n_cols", [5, 16, 37, 64, 100])
def test_rmsnorm_kernel_matches_the_formula_on_ragged_rows(n_cols):
    rng = np.random.default_rng(n_cols)
    x = rng.standard_normal((6, n_cols)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, n_cols).astype(np.float32)
    out = np.zeros_like(x)
    tl.launch(rmsnorm_kernel, (6,), x, w, out, n_cols, 1e-5, BLOCK=16)
    want = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-6)


def test_an_unmasked_access_outside_the_tensor_fails_loudly():
    x = np.ones((3, 20), dtype=np.float32)
    w, out = np.ones(20, dtype=np.float32), np.zeros((3, 20), dtype=np.float32)
    with pytest.raises(tl.OutOfBounds, match="accessed element 60 of `x_ptr`, which has 60"):
        tl.launch(rmsnorm_no_mask, (3,), x, w, out, 20, 1e-5, BLOCK=32)

    @tl.jit
    def before_start(x_ptr, n):
        tl.load(x_ptr + tl.arange(0, 4) - 1)  # element -1 must not wrap around

    with pytest.raises(tl.OutOfBounds, match="element -1"):
        tl.launch(before_start, (1,), x, 4)


def test_reading_past_a_row_inside_the_tensor_is_a_wrong_answer_or_a_race():
    # No tail mask, but the block stays inside the tensor: garbage from the next row.
    x = np.arange(1, 49, dtype=np.float32).reshape(2, 24)
    out = np.zeros_like(x)
    with pytest.raises(tl.DataRace, match="another program of this launch had already written"):
        tl.launch(rmsnorm_no_mask, (2,), x, np.ones(24, np.float32), out, 16, 1e-5, BLOCK=32)


def test_masked_off_elements_are_poison_without_other():
    @tl.jit
    def total(x_ptr, out_ptr, n, WITH_OTHER: tl.constexpr):
        offs = tl.arange(0, 8)
        if WITH_OTHER:
            x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        else:
            x = tl.load(x_ptr + offs, mask=offs < n)
        tl.store(out_ptr, tl.sum(x))

    x, out = np.ones(5, dtype=np.float32), np.zeros(1, dtype=np.float32)
    tl.launch(total, (1,), x, out, 5, WITH_OTHER=True)
    assert out[0] == 5.0
    tl.launch(total, (1,), x, out, 5, WITH_OTHER=False)
    assert np.isnan(out[0])

    @tl.jit
    def total_int(x_ptr, out_ptr, n):
        offs = tl.arange(0, 8)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n))

    codes, got = np.arange(5, dtype=np.uint8), np.zeros(8, dtype=np.uint8)
    tl.launch(total_int, (1,), codes, got, 5)
    assert got.tolist() == [0, 1, 2, 3, 4, 255, 255, 255]


def test_shifts_and_masks_unpack_four_bit_codes():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, 44).astype(np.uint32)
    words = np.zeros(6, dtype=np.uint32)
    for j, code in enumerate(codes):
        words[j // 8] |= code << np.uint32((j % 8) * 4)
    out = np.zeros(44, dtype=np.float32)
    tl.launch(unpack_kernel, (tl.cdiv(44, 16),), words, out, 44, BLOCK=16)
    assert out.tolist() == codes.astype(np.float32).tolist()


def test_type_rules_follow_the_triton_semantics_page():
    out = launch_probe(K=-7)
    # Integer // and % round toward zero, as in C.
    assert out[:8].tolist() == [-1, -1, 0, 0, 0, 0, 0, 1]
    assert out[8:16].tolist() == [-1, 0, -2, -1, 0, 1, 2, 0]
    assert out[16] == 800  # sum widens uint8 to uint32
    assert out[17] == (200 + 100) % 256  # uint8 + int scalar stays uint8
    assert out[18] == 4
    assert out[19] == -3  # constexpr arithmetic is Python's: -7 // 3 == -3
    assert out[20] == 10 and out[21] == 7

    f32, i32, u8, i1 = tl.float32, tl.int32, tl.uint8, tl.int1
    t = lambda ty: tl.tensor(np.ones(2), ty)
    assert (t(i32) + t(f32)).dtype is f32  # kind
    assert (t(tl.int16) + t(i32)).dtype is i32  # width
    assert (t(i32) + t(tl.uint32)).dtype is tl.uint32  # same width: unsigned
    assert (t(u8) + 300).dtype is u8  # a scalar of the same kind does not promote
    assert (t(tl.int16) * 4.0).dtype is f32  # a scalar of a higher kind does
    assert (t(i32) / t(i32)).dtype is f32  # integer division is float32
    # Division and modulo with a float narrower than float32 compute in float32.
    assert (t(tl.float16) / 2).dtype is f32 and (t(tl.float16) % 2.0).dtype is f32
    assert (t(tl.float16) * 2).dtype is tl.float16 and (t(tl.float64) / 2).dtype is tl.float64
    assert (t(i32) % t(i32)).dtype is i32
    assert (t(i32) < 3).dtype is i1 and (t(i1) & t(i1)).dtype is i1
    assert (2.0 * t(f32)).dtype is f32 and (np.float32(2) * t(f32)).dtype is f32
    assert tl.sum(t(tl.int8)).dtype is i32 and tl.sum(t(tl.float16)).dtype is tl.float16
    assert tl.where(t(i1), 1.0, 0.0).dtype is f32
    assert t(f32).to(tl.float16).dtype is tl.float16
    assert tl.cast(tl.tensor(np.float32(1.0), f32), tl.uint32, bitcast=True).value == 0x3F800000
    assert t(f32)[:, None].shape == (2, 1) and (t(f32)[:, None] * t(f32)[None, :]).shape == (2, 2)


def test_scalar_arguments_arrive_typed_and_constexprs_stay_python(monkeypatch):
    monkeypatch.setattr(tl, "_check_source", lambda fn: None)  # the kernel spies on itself
    seen = {}

    @tl.jit
    def kernel(x_ptr, n, big, eps, flag, BLOCK: tl.constexpr):
        seen.update(n=n, big=big, eps=eps, flag=flag, block=BLOCK, ptr=x_ptr)

    tl.launch(kernel, (1,), np.zeros(4, np.float32), 4, 2**40, 1e-5, True, BLOCK=64)
    assert seen["n"].dtype is tl.int32 and seen["big"].dtype is tl.int64
    assert seen["eps"].dtype is tl.float32 and seen["flag"].dtype is tl.int1
    assert seen["block"] == 64 and type(seen["block"]) is int
    assert seen["ptr"].element_ty is tl.float32
    assert kernel.constexprs == {"BLOCK"}


def test_shapes_are_compile_time_powers_of_two(monkeypatch):
    monkeypatch.setattr(tl, "_check_source", lambda fn: None)  # bodies are passed in

    def run(body):
        @tl.jit
        def kernel(x_ptr, n, BLOCK: tl.constexpr):
            body(n, BLOCK)

        tl.launch(kernel, (1,), np.zeros(4, np.float32), 4, BLOCK=12)

    with pytest.raises(tl.TritonShimError, match="power of two, got 12"):
        run(lambda n, block: tl.arange(0, block))
    with pytest.raises(tl.TritonShimError, match="compile-time constant"):
        run(lambda n, block: tl.arange(0, n))
    with pytest.raises(tl.TritonShimError, match="power of two"):
        run(lambda n, block: tl.zeros((block,), tl.float32))
    with pytest.raises(tl.TritonShimError, match="truth value of a block"):
        run(lambda n, block: bool(tl.arange(0, 4) < n))
    with pytest.raises(tl.TritonShimError, match="needs integer operands"):
        run(lambda n, block: tl.zeros((4,), tl.float32) // 2)
    with pytest.raises(tl.TritonShimError, match="division by zero"):
        run(lambda n, block: tl.arange(0, 4) // 0)
    with pytest.raises(tl.TritonShimError, match="needs a pointer"):
        run(lambda n, block: tl.load(tl.zeros((4,), tl.float32) + n))

    @tl.jit
    def float_offset(x_ptr, n):
        tl.load(x_ptr + tl.zeros((4,), tl.float32))

    with pytest.raises(tl.TritonShimError, match="offset only by integers"):
        tl.launch(float_offset, (1,), np.zeros(4, np.float32), 4)


def test_programs_must_not_write_the_same_element():
    @tl.jit
    def everyone_writes(out_ptr, n):
        tl.store(out_ptr + tl.arange(0, 4), 1.0)

    out = np.zeros(4, dtype=np.float32)
    tl.launch(everyone_writes, (1,), out, 4)  # one program may write twice
    with pytest.raises(tl.DataRace):
        tl.launch(everyone_writes, (2,), out, 4)


def test_a_two_dimensional_grid_and_num_programs():
    @tl.jit
    def cell(out_ptr, n_cols):
        r, c = tl.program_id(0), tl.program_id(1)
        tl.store(out_ptr + r * n_cols + c, r * 10 + c + tl.num_programs(1) * 100)

    out = np.zeros((3, 4), dtype=np.int32)
    info = tl.launch(cell, (3, 4), out, 4)
    assert info.programs == 12
    assert out.tolist() == [[400 + r * 10 + c for c in range(4)] for r in range(3)]


def test_jit_rejects_source_triton_could_not_compile():
    def uses_numpy(x_ptr, n):
        return np.sum(tl.load(x_ptr + tl.arange(0, 4)).value)

    def reaches_inside(x_ptr, n):
        x = tl.load(x_ptr + tl.arange(0, 4))
        x.value[0] = 1.0

    def builds_a_list(x_ptr, n):
        parts = [tl.load(x_ptr + i) for i in range(4)]  # noqa: F841

    def calls_len(x_ptr, n):
        return len(x_ptr)

    def outside_the_subset(x_ptr, n):
        return tl.compile_with_triton(x_ptr)

    for fn, what in (
        (uses_numpy, "`np.sum`"),
        (reaches_inside, "`.value`"),
        (builds_a_list, "ListComp"),
        (calls_len, "calling `len`"),
        (outside_the_subset, "not in the shim's subset"),
    ):
        with pytest.raises(tl.TritonShimError, match=what) as failure:
            tl.jit(fn)
        assert "cannot be used inside a Triton kernel" in str(failure.value)
        assert fn.__name__ in str(failure.value)


def test_launch_argument_errors_are_explained():
    x = np.zeros(8, dtype=np.float32)
    with pytest.raises(tl.TritonShimError, match="needs a kernel decorated with @tl.jit"):
        tl.launch(lambda: None, (1,))
    with pytest.raises(tl.TritonShimError, match="grid must be one to three"):
        tl.launch(add_kernel, (1, 1, 1, 1), x, x, x, 8, BLOCK_SIZE=8)
    with pytest.raises(tl.TritonShimError, match="not contiguous"):
        tl.launch(add_kernel, (1,), torch.zeros(4, 4).T, x, x, 8, BLOCK_SIZE=8)
    with pytest.raises(tl.TritonShimError, match="missing a required argument"):
        tl.launch(add_kernel, (1,), x, x, x, 8)
    with pytest.raises(tl.TritonShimError, match="launch it with tl.launch"):
        add_kernel(x, x, x, 8, 8)
    with pytest.raises(tl.TritonShimError, match="inside a kernel"):
        tl.program_id(0)
    with pytest.raises(tl.TritonShimError, match="no Triton dtype"):
        tl.launch(add_kernel, (1,), np.zeros(8, dtype=np.complex64), x, x, 8, BLOCK_SIZE=8)
    read_only = np.zeros(8, dtype=np.float32)
    read_only.setflags(write=False)
    with pytest.raises(tl.TritonShimError, match="read-only"):
        tl.launch(add_kernel, (1,), x, x, read_only, 8, BLOCK_SIZE=8)


def test_cuda_tensors_are_handed_to_real_triton(monkeypatch):
    """The route to the real compiler, with a stand-in `triton` module: the kernel
    function is rebuilt with `triton.language` in place of the shim. Real Triton has
    never run this."""
    language = types.ModuleType("triton.language")
    language.constexpr = type("constexpr", (), {})
    triton = types.ModuleType("triton")
    triton.language = language
    compiled = []

    class Jitted:
        def __init__(self, fn):
            self.fn = fn
            compiled.append(fn)

        def __getitem__(self, grid):
            return lambda *args, **kwargs: launched.append((grid, args, kwargs))

    launched = []
    triton.jit = Jitted
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)

    class OnCuda:
        device = types.SimpleNamespace(type="cuda")

    x = OnCuda()
    assert tl.launch(add_kernel, (4,), x, x, x, 100, BLOCK_SIZE=32) is None
    tl.launch(add_kernel, (4,), x, x, x, 100, BLOCK_SIZE=32)
    assert len(compiled) == 1 and len(launched) == 2  # compiled once, launched twice
    twin = compiled[0]
    assert twin.__code__ is add_kernel.fn.__code__  # the same source, untouched
    assert twin.__globals__["tl"] is language and add_kernel.fn.__globals__["tl"] is tl
    assert twin.__annotations__["BLOCK_SIZE"] is language.constexpr
    assert launched[0] == ((4,), (x, x, x, 100), {"BLOCK_SIZE": 32})
    del add_kernel._compiled


def test_cuda_tensors_without_triton_say_what_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)

    class OnCuda:
        device = types.SimpleNamespace(type="cuda")

    with pytest.raises(tl.TritonShimError, match="install `triton`"):
        tl.launch(add_kernel, (1,), OnCuda(), OnCuda(), OnCuda(), 8, BLOCK_SIZE=8)


def test_host_helpers():
    assert tl.cdiv(100, 32) == 4 and tl.cdiv(96, 32) == 3
    assert [tl.next_power_of_2(n) for n in (1, 2, 3, 17, 64)] == [1, 2, 4, 32, 64]


# -- what was re-read from the documentation for milestone 5 ---------------------------------


@tl.jit
def layer_norm_mean(X, Mean, stride, N, BLOCK_SIZE: tl.constexpr):
    # The first loop of the layer-norm tutorial's forward kernel, as written there: the
    # shape of tl.zeros is a list and dtype is passed by keyword.
    row = tl.program_id(0)
    X += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    tl.store(Mean + row, mean)


def test_the_layer_norm_tutorial_spelling_of_zeros_is_accepted():
    x = np.arange(1, 49, dtype=np.float32).reshape(2, 24)
    mean = np.zeros(2, dtype=np.float32)
    tl.launch(layer_norm_mean, (2,), x, mean, 24, 24, BLOCK_SIZE=16)
    np.testing.assert_allclose(mean, x.mean(axis=1), rtol=1e-6)

    # A list is a shape and nothing else: any other list is still refused.
    def list_elsewhere(x_ptr, n):
        offsets = [0, 1, 2, 3]  # noqa: F841

    def list_as_value(x_ptr, n):
        return tl.full((4,), [1.0], tl.float32)

    for fn in (list_elsewhere, list_as_value):
        with pytest.raises(tl.TritonShimError, match="`List` syntax"):
            tl.jit(fn)

    @tl.jit
    def keyword_shape(out_ptr, BLOCK: tl.constexpr):
        tl.store(out_ptr + tl.arange(0, BLOCK), tl.full(shape=[BLOCK], value=2.0, dtype=tl.float32))

    out = np.zeros(4, dtype=np.float32)
    tl.launch(keyword_shape, (1,), out, BLOCK=4)
    assert out.tolist() == [2.0] * 4


def test_arange_accepts_what_the_documented_wording_and_a_power_of_two_length_both_allow(
    monkeypatch,
):
    monkeypatch.setattr(tl, "_check_source", lambda fn: None)

    def run(start, end):
        seen = []

        @tl.jit
        def kernel(x_ptr):
            seen.append(tl.arange(start, end))

        tl.launch(kernel, (1,), np.zeros(4, np.float32))
        return seen[0].value.tolist()

    assert run(0, 4) == [0, 1, 2, 3]  # every tutorial: tl.arange(0, BLOCK)
    assert run(0, 1) == [0]
    assert run(4, 8) == [4, 5, 6, 7]  # start and end powers of two, and so is the length
    assert run(1, 2) == [1]
    # "start: Must be a power of two" (0 is what the tutorials use).
    with pytest.raises(tl.TritonShimError, match="start must be 0 or a power of two, got 3"):
        run(3, 7)
    # "end: Must be a power of two greater than start".
    with pytest.raises(tl.TritonShimError, match="end must be a power of two, got 12"):
        run(4, 12)
    with pytest.raises(tl.TritonShimError, match="greater than start"):
        run(4, 4)
    # Powers of two at both ends, but six elements: refused as a block shape.
    with pytest.raises(tl.TritonShimError, match="number of elements must be a power of two"):
        run(2, 8)


def test_shifts_the_documentation_does_not_define_are_refused(monkeypatch):
    monkeypatch.setattr(tl, "_check_source", lambda fn: None)

    def run(body, values, dtype):
        out = []

        @tl.jit
        def kernel(x_ptr):
            out.append(body(tl.load(x_ptr + tl.arange(0, 4))))

        tl.launch(kernel, (1,), np.asarray(values, dtype=dtype))
        return out[0]

    # Unsigned: zeros come in. Non-negative signed values shift the same way.
    got = run(lambda x: x >> 28, [0xF0000000, 0x80000000, 0x7FFFFFFF, 1], np.uint32)
    assert got.dtype is tl.uint32 and got.value.tolist() == [15, 8, 7, 0]
    assert run(lambda x: x >> 1, [8, 6, 1, 0], np.int32).value.tolist() == [4, 3, 0, 0]
    # A left shift may move a one into the sign bit (two's complement, as everywhere).
    got = run(lambda x: x << 28, [15, 8, 7, 0], np.int32)
    assert got.value.tolist() == [-(1 << 28), -(1 << 31), 7 << 28, 0]
    # Signed and negative: arithmetic or logical? Not documented, so not answered.
    with pytest.raises(tl.TritonShimError, match="does not say whether"):
        run(lambda x: x >> 28, [-(1 << 28), 0, 0, 0], np.int32)
    with pytest.raises(tl.TritonShimError, match="does not say whether"):
        run(lambda x: (x << 28) >> 28, [15, 8, 7, 0], np.int32)  # the bit-field idiom
    # Same width, mixed signedness: the result is unsigned (documented), so it shifts.
    got = run(lambda x: x.to(tl.uint32) >> x.to(tl.int32), [4, 2, 1, 0], np.int32)
    assert got.dtype is tl.uint32
    # A negative value on the left stays refused when the count is unsigned.
    with pytest.raises(tl.TritonShimError, match="does not say whether"):
        run(lambda x: x >> tl.full((4,), 1, tl.uint32), [-4, 2, 1, 0], np.int32)
    # A count outside [0, bits).
    for body in (lambda x: x >> 32, lambda x: x << 40, lambda x: x >> -1, lambda x: 1 << x):
        with pytest.raises(tl.TritonShimError, match="shift count outside"):
            run(body, [1, 2, 3, 33], np.uint32)
