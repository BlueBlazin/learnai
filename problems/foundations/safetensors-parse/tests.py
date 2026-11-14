"""Tests for safetensors-parse.

Synthetic files are written here, byte by byte, by `build` (so the expected arrays
are the ones that went in), plus one small file written by the `safetensors` package
(`fixtures/tiny.safetensors`). None of those cases needs a download. The last three
cases read the real 705 MB checkpoint through an mmap and compare every tensor with
what the `safetensors` package read from it (`fixtures/granite-4.0-350m.json`).

Arrays are compared by dtype, shape, and raw bytes, so NaN payloads and signed zeros
count and nothing is converted on the way.
"""

import json
import mmap
import struct

import numpy as np

from whetstone.judge import case, check

ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"

#: safetensors dtype name -> the numpy dtype read_tensor must return.
DTYPES = {
    "BOOL": "?",
    "U8": "u1",
    "I8": "i1",
    "U16": "<u2",
    "I16": "<i2",
    "U32": "<u4",
    "I32": "<i4",
    "U64": "<u8",
    "I64": "<i8",
    "F16": "<f2",
    "BF16": "<u2",
    "F32": "<f4",
    "F64": "<f8",
}


def build(tensors, metadata=None, *, layout=None, align=8, metadata_first=True):
    """The bytes of a safetensors file, and its tensor table.

    tensors: name -> (dtype name, array), in header order. layout: the order in which
    the data is stored (default: header order). align: the header is padded with
    spaces until the data starts at a multiple of `align` (1: no padding).
    """
    offsets, data = {}, b""
    for name in layout or list(tensors):
        raw = np.ascontiguousarray(tensors[name][1]).tobytes()
        offsets[name] = [len(data), len(data) + len(raw)]
        data += raw
    table = {
        name: {"dtype": kind, "shape": list(array.shape), "data_offsets": offsets[name]}
        for name, (kind, array) in tensors.items()
    }
    header = dict(table)
    if metadata is not None:
        header = {"__metadata__": metadata, **table} if metadata_first else {**table, "__metadata__": metadata}  # fmt: skip
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-(8 + len(text)) % align)
    return struct.pack("<Q", len(text)) + text + data, table


def arr(kind, values):
    return np.array(values, dtype=DTYPES[kind])


def as_mmap(ctx, data):
    """An anonymous memory map holding `data`, closed after the case."""
    mapped = mmap.mmap(-1, len(data))
    mapped[:] = data
    return mapped


def check_header(ctx, buf, table, metadata, data_start, label="read_header"):
    got = ctx.solution.read_header(buf)
    check.that(
        isinstance(got, tuple) and len(got) == 3,
        f"{label}: must return a tuple (tensors, metadata, data_start)",
        f"got: {type(got).__name__}",
    )
    tensors, got_metadata, got_start = got
    check.that(isinstance(tensors, dict), f"{label}: tensors must be a dict")
    check.equal(list(tensors), list(table), label=f"{label}: tensor names, in header order")
    check.equal(tensors, table, label=f"{label}: tensors")
    check.equal(got_metadata, metadata, label=f"{label}: metadata")
    check.that(
        type(got_start) is int and got_start == data_start,
        f"{label}: data_start is {got_start!r}, which is not where the data begins",
        f"expected {data_start}",
    )


def check_tensor(ctx, buf, name, kind, want, label=None):
    """read_tensor(buf, name) must be a view of `buf` with the right dtype, shape, bytes."""
    label = label or f"read_tensor(buf, {name!r})"
    got = ctx.solution.read_tensor(buf, name)
    check.that(
        isinstance(got, np.ndarray), f"{label}: returned a {type(got).__name__}, not an ndarray"
    )
    dtype = np.dtype(DTYPES[kind])
    check.that(
        got.dtype == dtype,
        f"{label}: got dtype {got.dtype}, which is not the dtype the statement gives for it",
        f"expected {dtype} for safetensors dtype {kind}",
    )
    check.shape(got, tuple(want.shape), label=label)
    same = np.ascontiguousarray(got).tobytes() == np.ascontiguousarray(want).tobytes()
    if not same:
        wrong = int(np.sum(np.ascontiguousarray(got).view(np.uint8) != want.view(np.uint8)))
        check.that(
            False,
            f"{label}: {wrong} of {want.nbytes} bytes differ from the stored tensor",
            f"expected =\n{want!r}\ngot =\n{got!r}",
        )
    if want.size:
        whole = np.frombuffer(buf, dtype=np.uint8)
        check.that(
            np.shares_memory(got, whole),
            f"{label}: the array is a copy; it must be a view of the buffer's memory",
        )
    return got


# ---------------------------------------------------------------------------------
# Synthetic files: no download needed.
# ---------------------------------------------------------------------------------

EXAMPLE = {
    "bias": ("BF16", arr("BF16", [0x3F80, 0xC020])),
    "ids": ("U8", arr("U8", [[1, 2, 3], [4, 5, 6]])),
}


@case("the worked example: a 170-byte file with a BF16 vector and a U8 matrix",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    data, table = build(EXAMPLE, {"format": "pt"})
    check.that(len(data) == 170 and data[:8] == bytes([0x98, 0, 0, 0, 0, 0, 0, 0]), "test bug")
    check_header(ctx, data, table, {"format": "pt"}, 160)
    check_tensor(ctx, data, "bias", "BF16", arr("BF16", [0x3F80, 0xC020]))
    check_tensor(ctx, data, "ids", "U8", arr("U8", [[1, 2, 3], [4, 5, 6]]))


def every_dtype(rng):
    """One tensor of every dtype, with values whose bytes are not palindromes."""
    f32 = rng.standard_normal((3, 5)).astype("<f4")
    return {
        "f32": ("F32", f32),
        "f64": ("F64", rng.standard_normal((2, 2, 3)).astype("<f8")),
        "f16": ("F16", rng.standard_normal(7).astype("<f2")),
        # bfloat16: the top 16 bits of each float32 (truncation is fine for a test).
        "bf16": ("BF16", (f32.view("<u4") >> 16).astype("<u2")),
        "u8": ("U8", rng.integers(0, 256, (4, 3)).astype("u1")),
        "i8": ("I8", rng.integers(-128, 128, 9).astype("i1")),
        "u16": ("U16", rng.integers(0, 2**16, (2, 6)).astype("<u2")),
        "i16": ("I16", rng.integers(-(2**15), 2**15, 5).astype("<i2")),
        "u32": ("U32", rng.integers(0, 2**32, (3, 2), dtype=np.uint64).astype("<u4")),
        "i32": ("I32", rng.integers(-(2**31), 2**31, 4).astype("<i4")),
        "u64": ("U64", rng.integers(0, 2**63, 3, dtype=np.uint64) * 2 + 1),
        "i64": ("I64", rng.integers(-(2**62), 2**62, (2, 2)).astype("<i8")),
        "bool": ("BOOL", rng.integers(0, 2, (3, 3)).astype("?")),
    }


# Rejects bf16_as_float16, big_endian_data, copies_data (in each kind of buffer), and
# fortran_order (the matrices are not square).
@case("all thirteen dtypes, from bytes, a bytearray, a memoryview, and an mmap: views, not copies",
      public=True, assets=[])  # fmt: skip
def all_dtypes(ctx):
    tensors = every_dtype(ctx.rng)
    data, table = build(tensors, {"format": "pt"})
    mapped = as_mmap(ctx, data)
    try:
        for buf in (data, bytearray(data), memoryview(data), mapped):
            kind_of_buf = type(buf).__name__
            check_header(ctx, buf, table, {"format": "pt"}, len(data) - sum(
                a.nbytes for _, a in tensors.values()), label=f"read_header({kind_of_buf})")  # fmt: skip
            for name, (kind, want) in tensors.items():
                got = check_tensor(
                    ctx, buf, name, kind, want, label=f"read_tensor({kind_of_buf}, {name!r})"
                )
                del got  # the mmap cannot be closed while a view of it is alive
    finally:
        try:
            mapped.close()
        except BufferError:  # a solution kept a view alive; the map dies with the process
            pass


# Rejects cumulative_offsets and offsets_from_file_start: the data is stored largest
# first, as real writers do, and the header is padded and has no metadata.
@case("data stored in a different order than the header lists it; a file without metadata",
      public=True, assets=[])  # fmt: skip
def shuffled_layout(ctx):
    rng = ctx.rng
    tensors = {
        "a.weight": ("F32", rng.standard_normal((2, 3)).astype("<f4")),
        "b.weight": ("I16", rng.integers(-99, 99, 5).astype("<i2")),
        "c.weight": ("F32", rng.standard_normal((4, 4)).astype("<f4")),
        "d.weight": ("U8", rng.integers(0, 256, 7).astype("u1")),
    }
    layout = ["c.weight", "a.weight", "d.weight", "b.weight"]
    for align in (8, 1, 64):
        data, table = build(tensors, None, layout=layout, align=align)
        start = len(data) - sum(a.nbytes for _, a in tensors.values())
        check_header(ctx, data, table, {}, start, label=f"read_header (data at byte {start})")
        for name, (kind, want) in tensors.items():
            check_tensor(ctx, data, name, kind, want)


@case("a file written by the safetensors package", public=True, assets=[])
def library_file(ctx):
    data = ctx.fixture("tiny.safetensors")
    want = ctx.fixture("tiny.npz")
    kinds = {"embed.weight": "F32", "norm.weight": "F16", "positions": "I64", "mask": "BOOL",
             "codes": "U32", "scale": "F64"}  # fmt: skip
    tensors, metadata, start = ctx.solution.read_header(data)
    check.equal(sorted(tensors), sorted(kinds), label="read_header: tensor names")
    check.equal(metadata, {"format": "np", "made_by": "make.py"}, label="read_header: metadata")
    check.equal(start, 448, label="read_header: data_start")
    for name, kind in kinds.items():
        check.equal(tensors[name]["dtype"], kind, label=f"dtype of {name}")
        check_tensor(ctx, data, name, kind, want[name].astype(DTYPES[kind]))


# Rejects metadata_as_tensor wherever the metadata sits in the header.
@case("__metadata__ is not a tensor, wherever it appears in the header", assets=[])
def metadata_is_not_a_tensor(ctx):
    tensors = {"w": ("F32", arr("F32", [[1.5, -2.0]])), "b": ("I32", arr("I32", [7]))}
    meta = {"format": "mlx", "note": "räksmörgås"}
    for first in (True, False):
        data, table = build(tensors, meta, metadata_first=first)
        check_header(ctx, data, table, meta, len(data) - 12)
        check.raises(KeyError, ctx.solution.read_tensor, data, "__metadata__",
                     label="read_tensor(buf, '__metadata__')")  # fmt: skip
    data, table = build(tensors, {})
    check_header(ctx, data, table, {}, len(data) - 12)


@case("scalars (shape []) and empty tensors (a dimension of 0)", assets=[])
def scalars_and_empty(ctx):
    tensors = {
        "step": ("I64", arr("I64", 12345678901)),
        "empty": ("F32", np.zeros((0,), dtype="<f4")),
        "scale": ("BF16", arr("BF16", 0x3E00)),
        "empty2d": ("U16", np.zeros((3, 0), dtype="<u2")),
        "tail": ("U8", arr("U8", [9])),
        "last_empty": ("F64", np.zeros((0, 2), dtype="<f8")),
    }
    data, table = build(tensors, {"format": "pt"})
    check_header(ctx, data, table, {"format": "pt"}, len(data) - 11)
    for name, (kind, want) in tensors.items():
        check_tensor(ctx, data, name, kind, want)


# Rejects big_endian_length on its own terms: the length needs three bytes.
@case("a header longer than 65,535 bytes", assets=[])
def long_header(ctx):
    tensors = {f"layer.{i}.weight": ("F32", ctx.randn(2, 2).astype("<f4")) for i in range(40)}
    meta = {"comment": "x" * 70_000}
    data, table = build(tensors, meta, layout=sorted(tensors, reverse=True))
    check_header(ctx, data, table, meta, len(data) - 40 * 16)
    for name in ("layer.0.weight", "layer.17.weight", "layer.39.weight"):
        check_tensor(ctx, data, name, "F32", tensors[name][1])


def corrupt(data, table, name, **changes):
    """The file `data` with the header entry of `name` changed."""
    header = {n: dict(info) for n, info in table.items()}
    header[name].update(changes)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    old = struct.unpack("<Q", data[:8])[0]
    return struct.pack("<Q", len(text)) + text + data[8 + old :]


# Rejects ignores_end_offset.
@case("malformed files raise ValueError; unknown names raise KeyError", assets=[])
def malformed(ctx):
    solution = ctx.solution
    tensors = {
        "w": ("F32", arr("F32", [[1, 2, 3], [4, 5, 6]])),
        "v": ("U16", arr("U16", [1, 2, 3, 4])),
    }
    data, table = build(tensors, {"format": "pt"})
    check.raises(KeyError, solution.read_tensor, data, "missing", label="an unknown tensor name")
    check.raises(ValueError, solution.read_header, b"", label="read_header of an empty buffer")
    check.raises(ValueError, solution.read_header, data[:7], label="read_header of 7 bytes")
    check.raises(ValueError, solution.read_header, data[:40],
                 label="read_header of a file cut off inside its header")  # fmt: skip
    # The header is intact but the data is cut short: the header still parses.
    cut = data[:-3]
    check_header(ctx, cut, table, {"format": "pt"}, len(data) - 32)
    check_tensor(ctx, cut, "w", "F32", tensors["w"][1])
    check.raises(ValueError, solution.read_tensor, cut, "v",
                 label="read_tensor of a tensor whose data runs past the end of the file")  # fmt: skip
    # Byte ranges that do not match the shape, in both directions.
    for name, bad in (("w", [0, 20]), ("w", [0, 28]), ("v", [24, 30]), ("v", [28, 24])):
        broken = corrupt(data, table, name, data_offsets=bad)
        check.raises(ValueError, solution.read_tensor, broken, name,
                     label="read_tensor with data_offsets that do not match the shape")  # fmt: skip
    broken = corrupt(data, table, "v", dtype="C64")
    check.raises(ValueError, solution.read_tensor, broken, "v", label="read_tensor of dtype C64")
    # The other tensor of a file with one bad entry is still readable.
    check_tensor(ctx, broken, "w", "F32", tensors["w"][1])


@case("random files: random dtypes, shapes, layouts, and padding", assets=[])
def random_files(ctx):
    rng = ctx.rng
    kinds = list(DTYPES)
    for _ in range(12):
        tensors = {}
        for i in range(int(rng.integers(1, 9))):
            kind = kinds[int(rng.integers(len(kinds)))]
            shape = tuple(int(n) for n in rng.integers(0, 5, size=rng.integers(0, 4)))
            raw = rng.integers(0, 256, int(np.prod(shape)) * np.dtype(DTYPES[kind]).itemsize)
            array = raw.astype("u1").view(DTYPES[kind]).reshape(shape)
            if kind == "BOOL":
                array = (raw % 2).astype("?").reshape(shape)
            tensors[f"t{i}.{kind.lower()}"] = (kind, array)
        layout = [str(n) for n in rng.permutation(list(tensors))]
        meta = None if rng.integers(2) else {"format": "pt"}
        data, table = build(tensors, meta, layout=layout, align=int(rng.choice([1, 8, 16])),
                            metadata_first=bool(rng.integers(2)))  # fmt: skip
        start = len(data) - sum(a.nbytes for _, a in tensors.values())
        check_header(ctx, bytearray(data), table, meta or {}, start)
        for name, (kind, want) in tensors.items():
            check_tensor(ctx, data, name, kind, want)


# ---------------------------------------------------------------------------------
# The real checkpoint: granite-4.0-350m, 704,786,224 bytes, read through an mmap.
# ---------------------------------------------------------------------------------


def open_checkpoint(ctx):
    """(mmap of model.safetensors, what the safetensors package read from it)."""
    fixture = ctx.fixture("granite-4.0-350m.json")
    check.that(fixture["revision"] == REVISION, "the fixture was made from another revision")
    handle = ctx.asset(ASSET, "model.safetensors").open("rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    handle.close()
    check.that(len(mapped) == fixture["file_bytes"], "the checkpoint on disk has another size")
    return mapped, fixture


@case("the real checkpoint: 226 BF16 tensors, their names and shapes", public=True)
def real_header(ctx):
    mapped, fixture = open_checkpoint(ctx)
    tensors, metadata, start = ctx.solution.read_header(mapped)
    check.equal(len(tensors), 226, label="number of tensors")
    check.equal(sorted(tensors), sorted(fixture["tensors"]), label="tensor names")
    check.equal(metadata, fixture["metadata"], label="metadata")
    for name, want in fixture["tensors"].items():
        check.equal(tensors[name]["dtype"], want["dtype"], label=f"dtype of {name}")
        check.equal(tensors[name]["shape"], want["shape"], label=f"shape of {name}")
    stored = sum(info["data_offsets"][1] - info["data_offsets"][0] for info in tensors.values())
    check.equal(
        start + stored, fixture["file_bytes"], label="data_start + the bytes of all tensors"
    )
    check.equal(start, 26416, label="data_start")


@case("the real checkpoint: the embedding table is a view of the mapped file", public=True)
def real_embedding(ctx):
    mapped, fixture = open_checkpoint(ctx)
    name = "model.embed_tokens.weight"
    got = ctx.solution.read_tensor(mapped, name)
    check.that(isinstance(got, np.ndarray), "read_tensor must return a numpy array")
    check.that(got.dtype == np.dtype("<u2"), f"{name}: got dtype {got.dtype}; BF16 is uint16")
    check.shape(got, (100352, 1024), label=name)
    check.that(
        np.shares_memory(got, np.frombuffer(mapped, dtype=np.uint8)),
        f"{name}: the array is a copy of 205 MB; it must be a view of the mapped file",
    )
    want = fixture["tensors"][name]
    flat = got.reshape(-1)
    samples = [int(flat[(flat.size - 1) * k // 7]) for k in range(8)]
    check.equal(samples, want["samples"], label=f"{name}: bit patterns at eight positions")
    # Row 100257 is <|end_of_text|>; its first weights, decoded from the bit patterns.
    row = (got[100257, :4].astype(np.uint32) << 16).view(np.float32)
    check.finite(row, label="the first weights of row 100257, decoded")


@case("the real checkpoint: every tensor matches what the safetensors package reads")
def real_every_tensor(ctx):
    mapped, fixture = open_checkpoint(ctx)
    whole = np.frombuffer(mapped, dtype=np.uint8)
    for name, want in fixture["tensors"].items():
        got = ctx.solution.read_tensor(mapped, name)
        check.that(got.dtype == np.dtype("<u2"), f"{name}: got dtype {got.dtype}")
        check.shape(got, tuple(want["shape"]), label=name)
        check.that(np.shares_memory(got, whole), f"{name}: the array is a copy, not a view")
        flat = got.reshape(-1)
        samples = [int(flat[(flat.size - 1) * k // 7]) for k in range(8)]
        check.equal(samples, want["samples"], label=f"{name}: bit patterns at eight positions")
        total = int(got.sum(dtype=np.uint64))
        check.that(
            total == want["sum"],
            f"{name}: the sum of the 16-bit patterns is not the checkpoint's",
            f"expected {want['sum']}, got {total}",
        )
