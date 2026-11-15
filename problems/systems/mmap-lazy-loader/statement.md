Open a safetensors checkpoint without loading it: read the header, and hand out each
tensor only when it is asked for, as an array that points into the memory-mapped file
wherever that is possible. Parsing the header is the `safetensors-parse` problem. The
work here is in what is read and when, and in the two cases where a view of the file
is not an array you can give to anyone: bfloat16, and a tensor that starts at an odd
offset.

## Specification

```python
class LazyCheckpoint:
    def __init__(self, file): ...
    def keys(self) -> list[str]: ...
    def get(self, name: str) -> np.ndarray: ...
    def load_subset(self, names) -> dict[str, np.ndarray]: ...
```

**The file.** `file` is a kit `CountingFile` opened on a safetensors file. It is a
read-only memory map that counts what is read from it:

- `len(file)` is the size of the file in bytes, and costs nothing;
- `file[a:b]` is a zero-copy `memoryview` of bytes `a` to `b - 1`, and counts as
  reading `b - a` bytes: `file.counters.bytes_read` grows by `b - a`. The memory of
  the view is the mapped file itself, so `np.frombuffer(file[a:b], dtype)` is an
  array over the file's pages without a copy.

Everything you read goes through `file`; it has no path to open a second time.

**The format** (as in `safetensors-parse`). Bytes 0 to 7 hold the header length $n$,
an unsigned 64-bit little-endian integer. Bytes 8 to $8 + n - 1$ are UTF-8 JSON, an
object that maps each tensor name to `{"dtype", "shape", "data_offsets": [begin,
end]}`; the key `"__metadata__"`, if present, is not a tensor. The tensor occupies
bytes `begin` to `end - 1` of the **data section**, which starts at byte $8 + n$ of
the file, as little-endian elements in row-major order.

**`LazyCheckpoint(file)`** reads the header and nothing else: when the constructor
returns, `bytes_read` is at most $8 + n$.

**`keys()`** returns the tensor names as a list, in the order of the header, without
`"__metadata__"`. It reads nothing.

**`get(name)`** returns the tensor as a numpy array with the shape given in the
header. The first call for a name reads exactly that tensor's `end - begin` bytes;
what it returns depends on the dtype and on where the tensor lies:

| stored as | returned |
|---|---|
| `BF16` | a **float32** array with the decoded values (new memory: numpy has no bfloat16) |
| any other dtype, at a file offset that is a multiple of its element size | a **view** of the mapped file with the dtype below: no copy |
| any other dtype, at any other file offset | a **copy** with the dtype below, in aligned memory |

| `dtype` | numpy | | `dtype` | numpy | | `dtype` | numpy |
|---|---|---|---|---|---|---|---|
| `BOOL` | `?` | | `U32` | `<u4` | | `F16` | `<f2` |
| `U8` | `u1` | | `I32` | `<i4` | | `F32` | `<f4` |
| `I8` | `i1` | | `U64` | `<u8` | | `F64` | `<f8` |
| `U16` | `<u2` | | `I64` | `<i8` | | | |
| `I16` | `<i2` | | | | | | |

- The file offset of a tensor is $8 + n + \texttt{begin}$. The map starts at a page
  boundary, so a view is an aligned array exactly when that offset is a multiple of
  the element size. Every array `get` returns has `array.flags.aligned == True`.
- A bfloat16 value is the upper 16 bits of the float32 with the same value: the
  pattern `b` decodes to the float32 whose bits are `b << 16`.
- **The result is cached.** A later `get(name)` returns the *same array object* and
  reads nothing.
- A tensor with zero elements is returned with its shape and dtype; there is
  nothing to read and nothing to share.
- `KeyError` if `name` is not a tensor of the file (`"__metadata__"` is not one).
  Nothing is read.

**`load_subset(names)`** takes an iterable of names and returns the dict
`{name: get(name)}`, with the keys in the order the names were given (a repeated name
appears once, at its first position). If **any** name is not a tensor of the file it
raises `KeyError` **before reading anything**: a load that fails halfway has paid for
tensors nobody will use.

## Example

`example.safetensors` has 291 bytes. Its header is $n = 256$ bytes of JSON (shown
here on several lines; in the file it is one line), so the data section starts at
byte 264:

```
{"__metadata__":{"format":"pt"},
 "w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},
 "flag":{"dtype":"U8","shape":[3],"data_offsets":[16,19]},
 "b":{"dtype":"BF16","shape":[2],"data_offsets":[19,23]},
 "v":{"dtype":"I16","shape":[2],"data_offsets":[23,27]}}
```

```python
ckpt = LazyCheckpoint(file)  # bytes_read: 264 (the header)
ckpt.keys()  # ["w", "flag", "b", "v"]; still 264

w = ckpt.get("w")  # float32 [[1, 2], [3, 4]]; bytes_read: 264 + 16 = 280
ckpt.get("w") is w  # True; still 280
ckpt.get("b")  # float32 [1.0, -2.5]; 284
ckpt.get("v")  # int16 [-2, 300]; 288
ckpt.load_subset(["flag", "w"])  # {"flag": uint8 [1, 0, 1], "w": w}; 291
```

- `w` starts at file offset 264, a multiple of 4: it is a view of the map.
- `flag` starts at offset 280; its elements are single bytes, so any offset is
  aligned: a view.
- `b` is BF16, the patterns `0x3F80` and `0xC020`: decoded to `[1.0, -2.5]` in new
  memory.
- `v` starts at offset $264 + 23 = 287$, which is odd, and its elements are 2 bytes:
  a view would be an unaligned array, so `v` is a copy.
- `load_subset(["flag", "w", "nope"])` on a fresh checkpoint raises `KeyError` with
  `bytes_read` still 264.

## Constraints

- Files are well formed: valid JSON, known dtypes, offsets that match the shapes and
  lie inside the file. Shapes have at most 4 dimensions. Test files are a few
  kilobytes; write the loader as if they were 100 GB.
- Arrays are compared exactly: dtype, shape, and bytes.
- **Cost.** At every moment `file.counters.bytes_read` is at most
  $8 + n$ plus the stored sizes (`end - begin`) of the distinct tensors requested so
  far. `file.read()`, `file[:]`, or a slice that spans several tensors is charged in
  full and exceeds it.
- The standard library and numpy are available.

## Why this matters

Starting a model server is mostly a matter of getting weights from disk to where they
are used. A loader that reads the file into memory and then builds tensors from it
holds the checkpoint twice at its peak and makes the process wait for every byte, also
for the layers that are about to be sent to another device or are not needed at all
(one LoRA adapter out of a hundred, the vision tower of a text-only request, one
shard's share of a model split across machines). Mapping the file and materializing
tensors on first use makes the open instantaneous, lets the operating system page
weights in as they are touched, and shares those pages between processes that open
the same file. The two exceptions in the table are where real loaders have had bugs:
bfloat16 checkpoints that come back as integers, and unaligned views that work on one
machine and crash, or crawl, in the extension that receives them.
