Implement `LazyCheckpoint`, a safetensors loader that reads only the file header on open and materializes each tensor on demand. Whenever possible, return a zero-copy NumPy view directly into the memory-mapped file, falling back to a new array only for `bfloat16` tensors or tensors stored at unaligned file offsets.

## Specification

```python
class LazyCheckpoint:
    def __init__(self, file): ...
    def keys(self) -> list[str]: ...
    def get(self, name: str) -> np.ndarray: ...
    def load_subset(self, names) -> dict[str, np.ndarray]: ...
```

**The file.** `file` is a read-only memory-mapped `CountingFile` wrapper around a safetensors file that counts bytes read:

- `len(file)` returns the file size in bytes and reads nothing.
- `file[a:b]` returns a zero-copy `memoryview` of bytes `a` through `b - 1` and adds `b - a` to `file.counters.bytes_read`. Because this view points directly to the mapped pages, `np.frombuffer(file[a:b], dtype)` constructs an array over the file's memory without copying.

Read all data through `file`. It has no path to open a second time.

**The format.** (Same as `safetensors-parse`.) Bytes 0 through 7 store the header length $n$ as an unsigned 64-bit little-endian integer. Bytes 8 through $8 + n - 1$ hold a UTF-8 JSON object mapping each tensor name to `{"dtype", "shape", "data_offsets": [begin, end]}`. If `"__metadata__"` is present, it is metadata rather than a tensor. The **data section** starts at byte $8 + n$, and each tensor occupies bytes `begin` through `end - 1` of that section as little-endian elements in row-major order.

**`LazyCheckpoint(file)`** reads only the header. When the constructor returns, `file.counters.bytes_read` must be at most $8 + n$.

**`keys()`** returns the tensor names as a list in header order, excluding `"__metadata__"`. It reads nothing.

**`get(name)`** returns the tensor as a NumPy array with the shape from the header. The first call for `name` reads exactly that tensor's `end - begin` bytes. What it returns depends on the dtype and file offset:

| stored as | returned |
|---|---|
| `BF16` | a **float32** array with the decoded values in new memory (NumPy has no bfloat16 dtype) |
| any other dtype, at a file offset that is a multiple of its element size | a **view** of the mapped file with the dtype below (no copy) |
| any other dtype, at any other file offset | a **copy** with the dtype below in aligned memory |

| `dtype` | numpy | | `dtype` | numpy | | `dtype` | numpy |
|---|---|---|---|---|---|---|---|
| `BOOL` | `?` | | `U32` | `<u4` | | `F16` | `<f2` |
| `U8` | `u1` | | `I32` | `<i4` | | `F32` | `<f4` |
| `I8` | `i1` | | `U64` | `<u8` | | `F64` | `<f8` |
| `U16` | `<u2` | | `I64` | `<i8` | | | |
| `I16` | `<i2` | | | | | | |

- A tensor's file offset is $8 + n + \texttt{begin}$. Because the memory map begins at a page boundary, a view is aligned if and only if this offset is a multiple of the element size. Every array returned by `get` must have `array.flags.aligned == True`.
- A `bfloat16` value holds the upper 16 bits of a `float32` value. A 16-bit pattern `b` decodes to the `float32` whose 32-bit pattern is `b << 16`.
- **Caching.** Cache each materialized tensor. Any subsequent `get(name)` returns the *same array object* (`is`) and reads nothing.
- A tensor with zero elements returns an array with its shape and dtype, reading nothing and sharing nothing.
- If `name` is not a tensor in the file (including `"__metadata__"`), raise `KeyError` without reading anything.

**`load_subset(names)`** takes an iterable of names and returns `{name: get(name)}`, keeping keys in the order the names were given (a repeated name appears once, at its first position). If **any** name is not a tensor in the file, raise `KeyError` **before reading anything** so a failing request never pays for tensors that will not be used.

## Example

`example.safetensors` is 291 bytes long. Its header is $n = 256$ bytes of JSON (formatted across multiple lines below for readability), so the data section starts at byte 264:

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

- `w` starts at file offset 264, which is a multiple of 4, so it is returned as a view of the map.
- `flag` starts at offset 280. Its elements are 1 byte each, so every offset is aligned and it is returned as a view.
- `b` is `BF16` with bit patterns `0x3F80` and `0xC020`, which decode to `[1.0, -2.5]` in new memory.
- `v` starts at offset $264 + 23 = 287$. Because 287 is odd and `int16` elements are 2 bytes, a view would be unaligned, so `v` is returned as a copy.
- Calling `load_subset(["flag", "w", "nope"])` on a fresh checkpoint raises `KeyError` with `bytes_read` still at 264.

## Constraints

- Files are well formed, with valid JSON, known dtypes, and offsets that match the shapes and lie within the file. Shapes have at most 4 dimensions. Test files are a few kilobytes, but write the loader as if they were 100 GB.
- Arrays are compared exactly by dtype, shape, and bytes.
- **Cost.** At every moment, `file.counters.bytes_read` must be at most $8 + n$ plus the stored sizes (`end - begin`) of the distinct tensors requested so far. Calling `file.read()`, `file[:]`, or slicing across multiple tensors is charged in full and violates this bound.
- The Python standard library and NumPy are available.

## Why this matters

Starting a model server is largely about moving weights from disk to where they are executed. A loader that reads an entire checkpoint into memory before constructing tensors doubles peak RAM usage and blocks startup on every byte, even for layers that belong on another device or are never needed (such as unused LoRA adapters, a vision tower during a text-only request, or other shards of a distributed model).

Memory-mapping the file and materializing tensors on first use makes opening instantaneous, lets the OS page weights in on demand, and shares physical pages across worker processes. The two copy cases in this problem are common sources of real-world bugs: returning raw `bfloat16` bits as integers, and handing unaligned views to C extensions or GPU upload paths that silently slow down or crash.
