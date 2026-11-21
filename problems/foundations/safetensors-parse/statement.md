Read a safetensors checkpoint without the `safetensors` package: parse the header of a
file held in memory, and hand out any tensor as a numpy array that points into the
file's bytes. The format is small enough to implement in twenty lines. The work is in
getting three offsets right and in never copying, because the file you are about to
open is 705 MB.

## Specification

```python
def read_header(buf) -> tuple[dict[str, dict], dict[str, str], int]: ...
def read_tensor(buf, name: str) -> np.ndarray: ...
```

`buf` is the whole file: a `bytes`, a `bytearray`, a `memoryview` of bytes, or an
`mmap.mmap`. A safetensors file has three parts.

![The three parts of a safetensors file. Every `data_offsets` pair counts from the start of the tensor data, byte $8 + n$.](assets/file-layout.svg)

1. **Bytes 0 to 7: the header length** $n$, an unsigned 64-bit **little-endian**
   integer.
2. **Bytes 8 to $8 + n - 1$: the header**, UTF-8 JSON. It is one object. Each key is
   a tensor name, and its value is an object with three fields:

   | field | meaning |
   |---|---|
   | `dtype` | one of the names in the table below |
   | `shape` | a list of non-negative integers; `[]` is a scalar |
   | `data_offsets` | `[begin, end]`: the tensor occupies bytes `begin` to `end - 1` **of the data section**, not of the file |

   One key is special: `"__metadata__"`, if present, is not a tensor. Its value is an
   object of free-form strings (`{"format": "pt"}`). It may appear anywhere in the
   header. The header may end in spaces (writers pad it so that the data starts at a
   multiple of 8); the spaces are part of the $n$ bytes.
3. **Bytes $8 + n$ to the end: the data section.** Tensors are stored as raw
   little-endian elements in row-major (C) order. They need not be stored in the
   order in which the header lists them.

**`read_header(buf)`** returns a tuple `(tensors, metadata, data_start)`:

- `tensors`: a dict from tensor name to its header entry, exactly as parsed (a dict
  with the keys `dtype`, `shape`, `data_offsets`), in the order of the header, without
  `"__metadata__"`.
- `metadata`: the `"__metadata__"` object, or `{}` if the file has none.
- `data_start`: the int $8 + n$.

It raises `ValueError` if `buf` has fewer than 8 bytes or fewer than $8 + n$ bytes.

**`read_tensor(buf, name)`** returns the tensor `name` as a numpy array:

- with the shape given in the header;
- with the numpy dtype below, little-endian whatever the machine;

  | `dtype` | numpy | | `dtype` | numpy | | `dtype` | numpy |
  |---|---|---|---|---|---|---|---|
  | `BOOL` | `?` | | `U32` | `<u4` | | `F16` | `<f2` |
  | `U8` | `u1` | | `I32` | `<i4` | | `F32` | `<f4` |
  | `I8` | `i1` | | `U64` | `<u8` | | `F64` | `<f8` |
  | `U16` | `<u2` | | `I64` | `<i8` | | `BF16` | `<u2` |
  | `I16` | `<i2` | | | | | | |

  numpy has no bfloat16, so a `BF16` tensor comes back as its 16-bit patterns,
  undecoded (decoding them is the `bf16-decode` problem);
- **as a view of `buf`, not a copy**: the array's memory is the buffer's memory
  (`np.shares_memory` is true). A tensor with zero elements is exempt, since it has no
  memory to share.

It raises

- `KeyError` if `name` is not a tensor of the file (`"__metadata__"` is not one);
- `ValueError` if the entry's `dtype` is not in the table, if `end - begin` is not
  the number of elements times the element size, or if the tensor's bytes end after
  the end of `buf`. A file with one bad entry still yields its other tensors.

Neither function modifies `buf`. Parse the bytes yourself: the `safetensors` package
and the kit's `SafetensorsFile` are not allowed. The standard library (`json`,
`struct`) and numpy are.

## Example

A complete file of 170 bytes, with a BF16 vector and a U8 matrix:

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

`bias` is at bytes `160 + 0` to `160 + 3` of the file. The pairs `80 3f` and `20 c0`
are the little-endian 16-bit integers `0x3F80` and `0xC020`, the bfloat16 patterns of
$1.0$ and $-2.5$. Read from the start of the file instead, offsets 0 to 3 would give
`98 00 00 00`: the header length.

## Constraints

- Everything is compared exactly: the header entries, the dtype, the shape, and the
  bytes of every array.
- The header is valid JSON whenever all $n$ bytes are present, and every entry has the
  three fields. Shapes have at most 4 dimensions.
- Synthetic files are a few kilobytes. The real one is `model.safetensors` of
  `ibm-granite/granite-4.0-350m`: 704,786,224 bytes, 226 tensors, all `BF16`, passed
  as a read-only `mmap`. Every one of its tensors is checked against what the
  `safetensors` package reads, and every array must be a view of the map.
- Calling `read_tensor` 226 times must stay well inside the time limit; parsing the
  header once per call is fine.

## Why this matters

safetensors is the format nearly every open model ships in, and its design goal is the
one this problem enforces: a loader should be able to map the file and use the tensors
where they lie, so that a 4 GB model "loads" in milliseconds and the operating system
pages weights in as they are touched. A loader that copies doubles peak memory at the
worst moment. One that gets the offset base wrong does not crash; it returns tensors
of the right shape filled with pieces of JSON and neighboring weights, and the model
produces fluent nonsense. Knowing the layout also lets you do what the libraries do
not offer: count parameters, or read one layer of a 100 GB checkpoint, without loading
anything.
