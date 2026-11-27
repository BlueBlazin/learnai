Read a safetensors checkpoint without using the `safetensors` package. Given a file held
in memory, parse its header and return any requested tensor as a NumPy array that views
the buffer directly without copying. Because real checkpoints can be hundreds of
megabytes (705 MB in the test suite), zero-copy reads and exact byte offsets are
essential.

## Specification

```python
def read_header(buf) -> tuple[dict[str, dict], dict[str, str], int]: ...
def read_tensor(buf, name: str) -> np.ndarray: ...
```

`buf` holds the entire file as a `bytes`, `bytearray`, `memoryview` of bytes, or
`mmap.mmap`. Every safetensors file has three contiguous sections: an 8-byte header
length $n$, an $n$-byte JSON header, and the raw tensor data starting at byte $8 + n$.

![The three parts of a safetensors file. Every `data_offsets` pair counts from the start of the tensor data, byte $8 + n$.](assets/file-layout.svg)

1. **Bytes 0 to 7 (header length).** An unsigned 64-bit **little-endian** integer $n$.
2. **Bytes 8 to $8 + n - 1$ (header).** A single UTF-8 JSON object. Each key is a tensor
   name, and its value is an object with three fields:

   | field | meaning |
   |---|---|
   | `dtype` | one of the dtype names in the table below |
   | `shape` | a list of non-negative integers (`[]` for a scalar) |
   | `data_offsets` | `[begin, end]`, meaning the tensor occupies bytes `begin` to `end - 1` **of the data section**, not of the file |

   One key is special. If `"__metadata__"` is present, it is not a tensor, and its value
   is an object of free-form strings such as `{"format": "pt"}`. It may appear anywhere
   in the header. The header may also end with trailing spaces (writers pad it so the
   data section starts on an 8-byte boundary), and those spaces are included in the $n$
   bytes.
3. **Bytes $8 + n$ to the end (data section).** Raw little-endian elements stored in
   row-major (C) order. Tensors do not have to appear in the same order as in the header.

**`read_header(buf)`** returns `(tensors, metadata, data_start)`:

- `tensors`: a dict mapping each tensor name to its parsed header entry (a dict with keys
  `dtype`, `shape`, and `data_offsets`), preserving header order and excluding
  `"__metadata__"`.
- `metadata`: the `"__metadata__"` dict, or `{}` if absent.
- `data_start`: the integer $8 + n$.

Raise `ValueError` if `buf` has fewer than 8 bytes or fewer than $8 + n$ bytes.

**`read_tensor(buf, name)`** returns tensor `name` as a NumPy array with the following
properties:

- Its shape matches the header's `shape`.
- Its NumPy dtype matches the little-endian table below, regardless of host endianness.

  | `dtype` | numpy | | `dtype` | numpy | | `dtype` | numpy |
  |---|---|---|---|---|---|---|---|
  | `BOOL` | `?` | | `U32` | `<u4` | | `F16` | `<f2` |
  | `U8` | `u1` | | `I32` | `<i4` | | `F32` | `<f4` |
  | `I8` | `i1` | | `U64` | `<u8` | | `F64` | `<f8` |
  | `U16` | `<u2` | | `I64` | `<i8` | | `BF16` | `<u2` |
  | `I16` | `<i2` | | | | | | |

  NumPy has no native `bfloat16`, so return `BF16` tensors as their raw 16-bit bit
  patterns (`<u2`), undecoded. Decoding them is covered in the `bf16-decode` problem.
- It must be a **view of `buf`, not a copy**, so `np.shares_memory(arr, buf)` is `True`.
  Tensors with zero elements are exempt because they have no backing bytes to share.

`read_tensor` must raise:

- `KeyError` if `name` is not a tensor in the file (`"__metadata__"` is not a tensor).
- `ValueError` if `dtype` is not in the table, if `end - begin` does not equal the
  element count times the element byte size, or if the tensor's byte range extends past
  the end of `buf`. A file with one malformed entry must still allow reading its other
  valid tensors.

Neither function may modify `buf`. Parse the bytes yourself without importing the
`safetensors` package or the kit's `SafetensorsFile`. The Python standard library
(`json`, `struct`, `math`, and so on) and NumPy are allowed.

## Example

Consider a complete 170-byte file containing a `BF16` vector and a `U8` matrix:

```
offset  bytes
     0  98 00 00 00 00 00 00 00                          n = 0x98 = 152
     8  {"__metadata__":{"format":"pt"},
        "bias":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},
        "ids":{"dtype":"U8","shape":[2,3],"data_offsets":[4,10]}}
        (146 bytes of JSON on one line, then 6 spaces)
   160  80 3f 20 c0                                      bias
   164  01 02 03 04 05 06                                ids
```

```python
tensors, metadata, data_start = read_header(buf)
# tensors == {"bias": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
#             "ids":  {"dtype": "U8", "shape": [2, 3], "data_offsets": [4, 10]}}
# metadata == {"format": "pt"}
# data_start == 160

read_tensor(buf, "bias")  # array([0x3F80, 0xC020], dtype=uint16)
read_tensor(buf, "ids")  # array([[1, 2, 3], [4, 5, 6]], dtype=uint8)
```

`bias` occupies bytes `160 + 0` through `160 + 3` of the file. The byte pairs `80 3f` and
`20 c0` decode as the little-endian 16-bit integers `0x3F80` and `0xC020`, which are the
`bfloat16` bit patterns for $1.0$ and $-2.5$. Reading from offset 0 of the file instead
would return `98 00 00 00`, which are the first four bytes of the header length.

## Constraints

- Header entries, dtypes, shapes, and array bytes are all compared for exact equality.
- Whenever all $n$ header bytes are present, the header is valid JSON and every tensor
  entry contains `dtype`, `shape`, and `data_offsets`. Shapes have at most 4 dimensions.
- Synthetic test files are a few kilobytes. The real test checkpoint is
  `model.safetensors` from `ibm-granite/granite-4.0-350m`, which is 704,786,224 bytes
  with 226 `BF16` tensors passed as a read-only `mmap`. Every tensor is checked against
  the `safetensors` package, and every returned array must be a view of the memory map.
- Calling `read_tensor` 226 times must finish well within the time limit. Parsing the
  header once per call is fast enough.

## Why this matters

Nearly every open-weights model ships in safetensors format. Its main goal is zero-copy
loading. A loader memory-maps the file and points tensors directly into the mapped pages,
so a 4 GB model opens in milliseconds and the operating system pages weights in only as
they are touched. Copying the buffer doubles peak memory at startup.

Getting the offset base wrong is a silent failure. The loader does not crash. Instead, it
returns tensors of the expected shape filled with JSON text and neighboring weights,
causing the model to generate garbage. Understanding the binary layout also lets you
inspect parameter counts or read a single layer from a 100 GB checkpoint without loading
the rest of the file.
