"""Tests for mlx-quantized-checkpoint-load.

Three kinds of file, and the format is the same in all of them:

- files assembled here byte by byte (`build`): the statement's example, a 2-bit layer,
  the error cases. The expected matrices are the ones the bytes were made from;
- two small checkpoints written by MLX itself (`fixtures/tiny-8bit.safetensors`, 8-bit
  codes in groups of 64 with bfloat16 scales; `tiny-4bit.safetensors`, 4-bit codes in
  groups of 32 with float16 and float32 scales), with every layer as `mx.dequantize`
  decodes it (`fixtures/tiny.npz`). None of these cases needs a download;
- the real `granite-4.2-3b-mlx-8bit` checkpoint, 3.9 GB, of which three layers are
  read and compared with what MLX reads (`fixtures/granite-4.2-3b-mlx-8bit.npz`).

Files on disk are handed over as a kit `CountingFile`, which charges every slice the
solution takes, so reading more than the header and the layer's three tensors shows.
"""

import json
import mmap
import struct

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import CountingFile, SafetensorsFile

FN = "load_quantized"
ASSET = "granite-4.2-3b-mlx-8bit"
REVISION = "7c630b526561db6b4641767c54595fe494176695"
TOL = {"rtol": 1e-6, "atol": 1e-7}  # the tolerance stated in Constraints
HEADER_READS = 3  # the byte limit allows the header to be read this many times


def bf16(values):
    """bfloat16 bit patterns of float32 values that bfloat16 represents exactly."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    assert not (bits & 0xFFFF).any(), "not exact in bfloat16"
    return (bits >> 16).astype("<u2")


def pack(codes, bits):
    """(out, in) codes -> (out, in * bits / 32) little-endian uint32 words, low bits first."""
    out_dim, in_dim = codes.shape
    per_word = 32 // bits
    grouped = codes.astype(np.uint64).reshape(out_dim, in_dim // per_word, per_word)
    shifts = np.arange(per_word, dtype=np.uint64) * np.uint64(bits)
    return (grouped << shifts).sum(axis=-1).astype("<u4")


def build(tensors, metadata=None, *, layout=None, pad_to=8, metadata_last=False):
    """The bytes of a safetensors file. tensors: name -> (dtype name, array), in header
    order; layout: the order of the data (default: header order)."""
    offsets, data = {}, b""
    for name in layout or list(tensors):
        raw = np.ascontiguousarray(tensors[name][1]).tobytes()
        offsets[name] = [len(data), len(data) + len(raw)]
        data += raw
    header = {
        name: {"dtype": kind, "shape": list(array.shape), "data_offsets": offsets[name]}
        for name, (kind, array) in tensors.items()
    }
    if metadata is not None:
        header = {**header, "__metadata__": metadata} if metadata_last else {"__metadata__": metadata, **header}  # fmt: skip
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-(8 + len(text)) % pad_to)
    return struct.pack("<Q", len(text)) + text + data


def layer(name, codes, scales, biases, bits, float_kind="BF16"):
    """The three tensors of one quantized layer, and the matrix they decode to."""
    codes, scales, biases = np.asarray(codes), np.asarray(scales), np.asarray(biases)
    group_size = codes.shape[1] // scales.shape[1]
    stored = {"BF16": bf16, "F16": lambda v: np.asarray(v, "<f2"), "F32": lambda v: np.asarray(v, "<f4")}[float_kind]  # fmt: skip
    tensors = {
        f"{name}.weight": ("U32", pack(codes, bits)),
        f"{name}.scales": (float_kind, stored(scales)),
        f"{name}.biases": (float_kind, stored(biases)),
    }
    s = np.repeat(np.asarray(scales, np.float32), group_size, axis=1)
    b = np.repeat(np.asarray(biases, np.float32), group_size, axis=1)
    return tensors, codes.astype(np.float32) * s + b


def check_matrix(got, want, label):
    check.that(
        isinstance(got, np.ndarray), f"{label}: returned a {type(got).__name__}, not an ndarray"
    )
    check.shape(got, want.shape, label=label)
    check.that(got.dtype == np.float32, f"{label}: got dtype {got.dtype}, not float32")
    check.allclose(got, want, **TOL, label=label)


def byte_limit(path, name):
    """What a loader may read for one layer: its three tensors, and the header three times."""
    file = SafetensorsFile(path)
    spans = [file.info(f"{name}.{part}")["data_offsets"] for part in ("weight", "scales", "biases")]
    return sum(end - begin for begin, end in spans) + HEADER_READS * file.data_start


def load_counted(ctx, path, name, bits, group_size, want, label, *, metric=False):
    with CountingFile(path) as f:
        got = ctx.solution.load_quantized(f, name, bits, group_size)
        got = np.array(got) if isinstance(got, np.ndarray) else got  # outlives the mapping
        read = f.counters.bytes_read
    check_matrix(got, want, label)
    limit = byte_limit(path, name)
    if metric:
        check.at_most("megabytes_read", round(read / 1e6, 4), round(limit / 1e6, 4), unit="MB")
    else:
        check.that(
            read <= limit,
            f"{label}: read {read} bytes of the file, which exceeds the limit of {limit} bytes "
            "(the layer's three tensors plus three times the header)",
        )
    return got


# ---------------------------------------------------------------------------------
# Small files: no download needed.
# ---------------------------------------------------------------------------------

EXAMPLE_CODES = np.array([[0, 1, 2, 3, 5, 10, 128, 255]])
EXAMPLE_WANT = np.array([[-1.0, -0.5, 0.0, 0.5, 0.75, -0.5, -30.0, -61.75]], dtype=np.float32)


@case("the worked example: 8 codes in two words, two groups of 4", public=True, assets=[])
def worked_example(ctx):
    tensors, want = layer("proj", EXAMPLE_CODES, [[0.5, -0.25]], [[-1.0, 2.0]], 8)
    check.that(bool((want == EXAMPLE_WANT).all()), "test setup: the example changed")
    data = build(tensors, {"format": "mlx"})
    got = ctx.solution.load_quantized(data, "proj", 8, 4)
    check_matrix(got, EXAMPLE_WANT, "load_quantized(example, 'proj', bits=8, group_size=4)")


@case("a checkpoint written by MLX: every 8-bit layer, groups of 64, bfloat16 scales",
      public=True, assets=[])  # fmt: skip
def mlx_8bit(ctx):
    path = ctx.fixture_path("tiny-8bit.safetensors")
    want = ctx.fixture("tiny.npz")
    names = [key.split("/", 1)[1] for key in want if key.startswith("tiny-8bit/")]
    check.that(len(names) == 11, "test setup: the fixture changed")
    for name in names:
        load_counted(ctx, path, name, 8, 64, want[f"tiny-8bit/{name}"],
                     f"load_quantized(tiny-8bit, {name!r}, bits=8, group_size=64)")  # fmt: skip


# Rejects reads_whole_file without the 3.9 GB file.
@case("only the header and the layer's three tensors are read", public=True, assets=[])
def reads_little(ctx):
    path = ctx.fixture_path("tiny-8bit.safetensors")
    name = "model.layers.0.mlp.gate_proj"
    load_counted(ctx, path, name, 8, 64, ctx.fixture("tiny.npz")[f"tiny-8bit/{name}"],
                 f"load_quantized(tiny-8bit, {name!r})", metric=True)  # fmt: skip


# Rejects high_nibble_first, assumes_8_bits, assumes_groups_of_64, f16_as_bf16.
@case("a checkpoint written by MLX: 4-bit codes, groups of 32, float16 and float32 scales",
      public=True, assets=[])  # fmt: skip
def mlx_4bit(ctx):
    path = ctx.fixture_path("tiny-4bit.safetensors")
    want = ctx.fixture("tiny.npz")
    names = [key.split("/", 1)[1] for key in want if key.startswith("tiny-4bit/")]
    check.that(len(names) == 6, "test setup: the fixture changed")
    for name in names:
        load_counted(ctx, path, name, 4, 32, want[f"tiny-4bit/{name}"],
                     f"load_quantized(tiny-4bit, {name!r}, bits=4, group_size=32)")  # fmt: skip


# Rejects missing_scales_ignored.
@case("KeyError for a layer that is not in the file, and for a tensor that is not quantized",
      assets=[])  # fmt: skip
def missing_parts(ctx):
    with CountingFile(ctx.fixture_path("tiny-8bit.safetensors")) as f:
        for name in ("model.layers.2.self_attn.q_proj", "model.norm", "__metadata__",
                     "model.layers.1.self_attn.k_proj.weight"):  # fmt: skip
            check.raises(KeyError, ctx.solution.load_quantized, f, name, 8, 64,
                         label="load_quantized(a name without the three tensors)")  # fmt: skip


@case("ValueError when bits or group_size do not fit the stored shapes", assets=[])
def wrong_settings(ctx):
    with CountingFile(ctx.fixture_path("tiny-8bit.safetensors")) as f:
        name = "model.layers.0.self_attn.q_proj"
        for bits, group_size in ((4, 64), (8, 32), (8, 128), (2, 64)):
            check.raises(ValueError, ctx.solution.load_quantized, f, name, bits, group_size,
                         label="load_quantized(settings that do not fit the file)")  # fmt: skip
    codes = np.arange(16).reshape(2, 8)
    tensors, _ = layer("a", codes, [[1.0, 2.0], [0.5, 4.0]], [[0.0, 1.0], [0.0, 0.0]], 8)
    tensors["a.weight"] = ("I32", tensors["a.weight"][1].astype("<i4"))
    check.raises(ValueError, ctx.solution.load_quantized, build(tensors), "a", 8, 4,
                 label="load_quantized(a .weight that is not U32)")  # fmt: skip


@case("2-bit codes in a memory map; data stored out of header order; metadata last", assets=[])
def two_bits(ctx):
    rng = ctx.rng
    codes = rng.integers(0, 4, (5, 64))
    scales = (rng.integers(1, 9, (5, 4)) / 8.0) * rng.choice([-1.0, 1.0], (5, 4))
    biases = rng.integers(-16, 17, (5, 4)) / 4.0
    tensors, want = layer("blocks.3.proj", codes, scales, biases, 2)
    other, other_want = layer("blocks.30.proj", rng.integers(0, 4, (2, 32)), [[0.5], [2.0]],
                              [[1.0], [-1.0]], 2, float_kind="F32")  # fmt: skip
    tensors.update(other)
    names = list(tensors)
    data = build(tensors, {"format": "mlx"}, layout=names[::-1], pad_to=1, metadata_last=True)
    mapped = mmap.mmap(-1, len(data))
    mapped[:] = data
    ctx.defer(mapped.close)
    got = ctx.solution.load_quantized(mapped, "blocks.3.proj", 2, 16)
    check_matrix(np.array(got), want, "load_quantized(2-bit layer)")
    got = ctx.solution.load_quantized(mapped, "blocks.30.proj", 2, 32)
    check_matrix(np.array(got), other_want, "load_quantized(2-bit layer, one group per row)")


# ---------------------------------------------------------------------------------
# granite-4.2-3b-mlx-8bit: 927 tensors in 3.9 GB.
# ---------------------------------------------------------------------------------


def real(ctx, name, *, hidden):
    fixture = ctx.fixture("granite-4.2-3b-mlx-8bit.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    path = ctx.asset(ASSET, "model.safetensors")
    what = "a layer" if hidden else repr(name)
    label = f"load_quantized(granite-4.2-3b, {what}, bits=8, group_size=64)"
    with CountingFile(path) as f:
        got = ctx.solution.load_quantized(f, name, 8, 64)
        read = f.counters.bytes_read
    check.that(isinstance(got, np.ndarray), f"{label}: returned a {type(got).__name__}")
    check.shape(got, tuple(int(n) for n in fixture[f"shape/{name}"]), label=label)
    check.that(got.dtype == np.float32, f"{label}: got dtype {got.dtype}, not float32")
    rows = [int(r) for r in fixture["rows"]]
    check.allclose(got[rows], fixture[f"rows/{name}"], **TOL, label=f"{label}: rows 0 and 300")
    wide = got.astype(np.float64)
    r, c = np.indices(got.shape)
    weight = ((r * 31 + c * 17) % 97 + 1).astype(np.float64)
    sums = np.array([wide.sum(), np.abs(wide).sum(), (wide * weight).sum()])
    scale = np.abs(wide).sum() * 97
    check.that(
        bool(np.all(np.abs(sums - fixture[f"checks/{name}"]) <= 1e-6 * scale)),
        f"{label}: rows 0 and 300 are right, but sums over the whole matrix are not",
        f"expected [sum, sum of |w|, position-weighted sum] = {fixture[f'checks/{name}']}\n"
        f"got {sums}",
    )
    check.at_most("megabytes_read", round(read / 1e6, 3), round(byte_limit(path, name) / 1e6, 3),
                  unit="MB")  # fmt: skip


@case("Granite 4.2 3B, 8-bit: model.layers.0.self_attn.k_proj, 1.4 MB of a 3.9 GB file",
      public=True)  # fmt: skip
def real_first_layer(ctx):
    real(ctx, "model.layers.0.self_attn.k_proj", hidden=False)


@case("Granite 4.2 3B, 8-bit: a value projection of the last layer")
def real_last_layer(ctx):
    real(ctx, "model.layers.39.self_attn.v_proj", hidden=True)


# Rejects offsets_in_int32: all three tensors of this layer lie beyond byte 2**31.
@case("Granite 4.2 3B, 8-bit: a layer stored in the second half of the file")
def real_far_layer(ctx):
    real(ctx, "model.layers.20.self_attn.k_proj", hidden=True)


@case("Granite 4.2 3B, 8-bit: model.norm has a .weight but is not quantized: KeyError")
def real_not_quantized(ctx):
    with CountingFile(ctx.asset(ASSET, "model.safetensors")) as f:
        check.raises(KeyError, ctx.solution.load_quantized, f, "model.norm", 8, 64,
                     label="load_quantized(a tensor that is not quantized)")  # fmt: skip
