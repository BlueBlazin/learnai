"""Tests for mmap-lazy-loader.

Every case opens a small safetensors file through the kit's `CountingFile` and checks
two things after each step: the array that came back (dtype, shape, bytes, whether it
is a view of the map, whether it is aligned, whether it is the cached object), and
`bytes_read` against the statement's budget, the header plus the distinct tensors
requested so far. The expected arrays come from the kit's own reader, `SafetensorsFile`.

The file is handed over behind the kit's `Guarded` (`File`): a solution may slice it,
take its length, and read it as a file. Its counters and its private mapping are
refused, in the constructor and in every later call, so the count cannot be written.

Two files are committed (`fixtures/example.safetensors`, `fixtures/mini.safetensors`,
written by `fixtures/make.py`); the hidden cases also write random files to a
temporary directory.
"""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from learnai.judge import case, check
from learnai.kit import (
    CountingFile,
    Guarded,
    SafetensorsFile,
    bf16_to_f32,
    check_guard,
    guard,
)

#: safetensors dtype name -> numpy dtype of the stored elements.
STORED = {
    "BOOL": "?", "U8": "u1", "I8": "i1", "U16": "<u2", "I16": "<i2", "U32": "<u4",
    "I32": "<i4", "U64": "<u8", "I64": "<i8", "F16": "<f2", "BF16": "<u2", "F32": "<f4",
    "F64": "<f8",
}  # fmt: skip


class File(Guarded, CountingFile):
    """The file a solution gets: slices, `len`, and reads, all counted."""

    label = "file"


def build(tensors, metadata=None, *, layout=None, align=8):
    """The bytes of a safetensors file. tensors: name -> (dtype name, array), in
    header order; layout: the order in which the data is stored."""
    offsets, data = {}, b""
    for name in layout or list(tensors):
        raw = np.ascontiguousarray(tensors[name][1]).tobytes()
        offsets[name] = [len(data), len(data) + len(raw)]
        data += raw
    header = {} if metadata is None else {"__metadata__": metadata}
    for name, (kind, array) in tensors.items():
        header[name] = {"dtype": kind, "shape": list(array.shape), "data_offsets": offsets[name]}
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-(8 + len(text)) % align)
    return struct.pack("<Q", len(text)) + text + data


def whole_file(file):
    """The mapped bytes as a uint8 array, without charging the solution for them."""
    before = file.counters.snapshot()
    whole = np.frombuffer(file[0 : len(file)], dtype=np.uint8)
    file.counters.reset()
    for name, value in before.items():
        file.counters.add(name, value)
    return whole


class Session:
    """One checkpoint opened by the solution, with the bookkeeping of the budget."""

    def __init__(self, ctx, path):
        self.ctx = ctx
        self.truth = SafetensorsFile(path)
        self.file = File(path)
        self.file.path = "<the counting file>"  # everything must go through `file`
        ctx.defer(self.file.close)
        self.header_bytes = self.truth.data_start
        self.requested: set[str] = set()
        self.seen: dict[str, np.ndarray] = {}
        self.ckpt = self.call(ctx.solution.LazyCheckpoint, self.file)
        self.check_budget("LazyCheckpoint(file)")

    def call(self, fn, *args):
        """`fn(*args)` of the solution, with the guard armed (the checkpoint keeps the file)."""
        guard(self.ctx)
        try:
            return fn(*args)
        finally:
            check_guard()

    def stored_bytes(self, name):
        begin, end = self.truth.info(name)["data_offsets"]
        return end - begin

    def budget(self):
        return self.header_bytes + sum(self.stored_bytes(name) for name in self.requested)

    def check_budget(self, after, *, metric=False):
        read, limit = self.file.counters.bytes_read, self.budget()
        if metric:
            check.at_most("bytes_read", read, limit, unit="bytes")
        check.that(
            read <= limit,
            f"after {after}: {read} bytes of the file have been read, which exceeds the "
            f"limit of {limit} bytes (the header plus the tensors requested so far)",
            f"header: {self.header_bytes} bytes; requested so far: {sorted(self.requested)}",
        )

    def expected(self, name):
        info = self.truth.info(name)
        raw = self.truth.raw(name)
        if info["dtype"] == "BF16":
            return bf16_to_f32(raw).reshape(info["shape"]), np.dtype(np.float32)
        return raw, np.dtype(STORED[info["dtype"]])

    def check_array(self, name, got, label):
        info = self.truth.info(name)
        want, dtype = self.expected(name)
        check.that(
            isinstance(got, np.ndarray), f"{label}: returned a {type(got).__name__}, not an ndarray"
        )
        check.that(
            got.dtype == dtype,
            f"{label}: got dtype {got.dtype}, expected dtype {dtype} (the tensor is stored as "
            f"{info['dtype']})",
            f"expected {dtype} for a tensor stored as {info['dtype']}",
        )
        check.shape(got, tuple(info["shape"]), label=label)
        if np.ascontiguousarray(got).tobytes() != np.ascontiguousarray(want).tobytes():
            wrong = int(np.sum(np.ascontiguousarray(got).view(np.uint8) != want.view(np.uint8)))
            check.that(
                False,
                f"{label}: {wrong} of {want.nbytes} bytes differ from the stored tensor",
                f"expected =\n{want!r}\ngot =\n{got!r}",
            )
        check.that(
            bool(got.flags.aligned),
            f"{label}: the array is not aligned (array.flags.aligned is False); a tensor at "
            "an unaligned file offset must be copied",
        )
        if name in self.seen:
            check.that(
                got is self.seen[name],
                f"{label}: a second request returned a different array object; the first "
                "result must be cached",
            )
        self.seen[name] = got
        offset = self.truth.data_start + info["data_offsets"][0]
        mappable = info["dtype"] != "BF16" and offset % dtype.itemsize == 0
        if want.size and mappable:
            check.that(
                np.shares_memory(got, whole_file(self.file)),
                f"{label}: the array is a copy; this tensor must be a view of the mapped file",
                f"stored as {info['dtype']} at file offset {offset}",
            )

    def get(self, name, *, metric=False):
        label = f"get({name!r})"
        got = self.call(self.ckpt.get, name)
        self.requested.add(name)
        self.check_array(name, got, label)
        self.check_budget(label, metric=metric)
        return got

    def load_subset(self, names, *, metric=False):
        label = f"load_subset({list(names)!r})"
        got = self.call(self.ckpt.load_subset, names)
        self.requested.update(names)
        check.that(isinstance(got, dict), f"{label}: returned a {type(got).__name__}, not a dict")
        check.equal(list(got), list(dict.fromkeys(names)), label=f"{label}: keys, in order")
        for name, array in got.items():
            self.check_array(name, array, f"{label}[{name!r}]")
        self.check_budget(label, metric=metric)
        return got

    def check_keys(self):
        before = self.file.counters.bytes_read
        keys = self.call(self.ckpt.keys)
        check.equal(keys, self.truth.names(), label="keys()")
        check.that(
            self.file.counters.bytes_read == before, "keys() read from the file; it must not"
        )

    def check_key_error(self, call, *args, label):
        before = self.file.counters.bytes_read
        check.raises(KeyError, self.call, call, *args, label=label)
        read = self.file.counters.bytes_read - before
        check.that(
            read == 0,
            f"{label}: {read} bytes were read before the KeyError; nothing may be read",
        )


def temp_file(ctx, data):
    """`data` written to a file in a temporary directory that is removed after the case."""
    directory = tempfile.TemporaryDirectory()
    ctx.defer(directory.cleanup)
    path = Path(directory.name) / "model.safetensors"
    path.write_bytes(data)
    return path


def bf16(values):
    return (np.asarray(values, dtype="<f4").view("<u4") >> 16).astype("<u2")


# ---------------------------------------------------------------------------------
# The committed files.
# ---------------------------------------------------------------------------------


@case("the worked example: 264 header bytes, then 16, 4, 4, and 3 bytes as tensors are used",
      public=True)  # fmt: skip
def worked_example(ctx):
    s = Session(ctx, ctx.fixture_path("example.safetensors"))
    check.that(s.header_bytes == 264 and len(s.file) == 291, "test bug")
    s.check_keys()
    w = s.get("w")
    check.equal(w.tolist(), [[1.0, 2.0], [3.0, 4.0]], label="get('w')")
    check.that(s.get("w") is w, "get('w') twice: the second call must return the same array")
    check.equal(s.get("b").tolist(), [1.0, -2.5], label="get('b')")
    check.equal(s.get("v").tolist(), [-2, 300], label="get('v')")
    s.load_subset(["flag", "w"], metric=True)


# Rejects reads_whole_file and eager_load: nothing but the header may be read here.
@case("opening reads only the header; keys() reads nothing", public=True)
def open_reads_header(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    s.check_keys()
    s.check_keys()
    check.at_most("bytes_read", s.file.counters.bytes_read, s.header_bytes, unit="bytes")


# Rejects offsets_from_byte_8, copies_data, and reads_whole_file.
@case("one tensor of a checkpoint: a view of the map, and only its bytes are read",
      public=True)  # fmt: skip
def one_tensor(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    s.get("embed.weight", metric=True)
    s2 = Session(ctx, ctx.fixture_path("mini.safetensors"))
    s2.get("layers.0.mlp.weight")
    s2.get("mask", metric=True)


# Rejects bf16_as_uint16.
@case("BF16 tensors come back as decoded float32", public=True)
def bf16_decoded(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    for name in ("layers.0.attn.weight", "layers.1.norm.weight", "layers.0.norm.weight"):
        s.get(name)
    s.check_budget("three BF16 tensors", metric=True)


# Rejects unaligned_view: these three lie at odd file offsets.
@case("tensors at unaligned offsets come back as aligned copies", public=True)
def unaligned_offsets(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    for name in ("layers.1.mlp.weight", "positions", "step", "layers.1.attn.weight"):
        s.get(name)
    s.check_budget("four tensors at odd offsets", metric=True)


# Rejects no_cache: the second request must cost nothing and return the same object.
@case("a tensor is read once: later requests return the cached array and read nothing")
def cached(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    names = ["layers.1.mlp.weight", "embed.weight", "layers.0.attn.weight", "step"]
    for name in names:
        s.get(name)
    for name in [*names, *names]:
        s.get(name)
    s.load_subset(names)
    s.load_subset(["embed.weight", "embed.weight", "mask", "embed.weight"], metric=True)


# Rejects subset_reads_before_checking.
@case("unknown names: KeyError, and load_subset reads nothing before raising")
def unknown_names(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    s.check_key_error(s.ckpt.get, "layers.2.attn.weight", label="get(an unknown name)")
    s.check_key_error(s.ckpt.get, "__metadata__", label="get('__metadata__')")
    s.check_key_error(
        s.ckpt.load_subset,
        ["embed.weight", "positions", "layers.9.mlp.weight"],
        label="load_subset(two tensors and an unknown name)",
    )
    s.check_key_error(
        s.ckpt.load_subset, ["mask", "__metadata__"], label="load_subset(['mask', '__metadata__'])"
    )
    s.load_subset(["embed.weight", "positions"], metric=True)


@case("a scalar, a tensor without elements, an empty subset, and a generator of names")
def odd_tensors(ctx):
    s = Session(ctx, ctx.fixture_path("mini.safetensors"))
    check.equal(s.load_subset([]), {}, label="load_subset([])")
    s.get("empty")
    check.equal(int(s.get("step")), 123456, label="get('step')")
    names = ["positions", "empty", "step"]
    got = s.call(s.ckpt.load_subset, (name for name in names))
    s.requested.update(names)
    check.equal(list(got), names, label="load_subset(a generator): keys, in order")
    s.check_budget("load_subset(a generator)", metric=True)


# ---------------------------------------------------------------------------------
# Files written by the tests.
# ---------------------------------------------------------------------------------


# Rejects offsets_from_byte_8 on its own terms: with no padding and no metadata the
# header is short, and the data is stored in reverse order.
@case("a file without metadata or header padding, data stored in reverse order")
def unpadded_file(ctx):
    rng = ctx.rng
    tensors = {
        "a": ("F32", rng.standard_normal((3, 2)).astype("<f4")),
        "b": ("BF16", bf16(rng.standard_normal((2, 5)))),
        "c": ("U8", rng.integers(0, 256, 5).astype("u1")),
        "d": ("F64", rng.standard_normal(3).astype("<f8")),
        "e": ("I16", rng.integers(-999, 999, (2, 2)).astype("<i2")),
    }
    for align in (1, 8, 64):
        data = build(tensors, None, layout=["e", "d", "c", "b", "a"], align=align)
        s = Session(ctx, temp_file(ctx, data))
        s.check_keys()
        s.get("d")
        s.load_subset(["a", "e"])
        s.get("b")
        s.get("c", metric=True)


@case("random files: every dtype, random shapes, layouts, and request orders")
def random_files(ctx):
    rng = ctx.rng
    kinds = list(STORED)
    for _ in range(8):
        tensors = {}
        for i in range(int(rng.integers(2, 10))):
            kind = kinds[int(rng.integers(len(kinds)))]
            shape = tuple(int(n) for n in rng.integers(0, 5, size=rng.integers(0, 4)))
            size = int(np.prod(shape)) * np.dtype(STORED[kind]).itemsize
            raw = rng.integers(0, 256, size).astype("u1")
            array = (raw % 2).astype("?") if kind == "BOOL" else raw.view(STORED[kind])
            tensors[f"t{i}.{kind.lower()}"] = (kind, array.reshape(shape))
        layout = [str(n) for n in rng.permutation(list(tensors))]
        meta = None if rng.integers(2) else {"format": "pt"}
        data = build(tensors, meta, layout=layout, align=int(rng.choice([1, 8, 16])))
        s = Session(ctx, temp_file(ctx, data))
        s.check_keys()
        names = list(tensors)
        for step in range(6):
            picked = [names[int(j)] for j in rng.integers(0, len(names), size=rng.integers(1, 4))]
            if step % 2:
                s.load_subset(picked)
            else:
                s.get(picked[0])
        s.check_budget("six random requests", metric=True)
