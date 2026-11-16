Open a real quantized checkpoint, the 8-bit MLX build of IBM's Granite 4.2 3B, and
recover the float weights of one layer from it by hand: find three tensors in a
3.9 GB safetensors file, unpack the codes from 32-bit words, decode the bfloat16
scales and biases, and apply them group by group. The file is far larger than what
you need from it, so the second half of the exercise is to read only that.

## Specification

```python
def load_quantized(buf, name, bits, group_size) -> np.ndarray: ...
```

- `buf`: the whole file, as an object that supports `len(buf)` and slicing:
  `buf[a:b]` returns bytes `a` to `b - 1` as a bytes-like object (`bytes` or a
  `memoryview`) that `np.frombuffer`, `bytes(...)`, and `int.from_bytes` accept. It
  may be a `bytes`, an `mmap.mmap`, or the kit's
  [`CountingFile`](../../../python/src/learnai/kit/mlxtools.py), a memory-mapped
  file that counts the bytes of every slice taken from it. Use slices only.
- `name`: the layer, as the model names it, without a suffix: for example
  `"model.layers.0.self_attn.k_proj"`.
- `bits`: the bits per code: 2, 4, or 8. `group_size`: the number of consecutive
  inputs that share one scale and one bias. Both come from the `quantization` entry
  of the model's `config.json` (`{"group_size": 64, "bits": 8}` for this checkpoint).
- Returns the layer's weight matrix, a float32 numpy array of shape `(out, in)`.

**The file** is a safetensors file, as in `safetensors-parse`: bytes 0 to 7 hold the
header length $n$ as an unsigned 64-bit little-endian integer; bytes 8 to $8 + n - 1$
are a JSON object that maps each tensor name to `{"dtype", "shape", "data_offsets":
[begin, end]}`; the tensor occupies bytes `begin` to `end - 1` **of the data
section**, which starts at byte $8 + n$. Tensors are little-endian, in row-major
order, and are not stored in any particular order. The key `"__metadata__"` is not a
tensor.

**A quantized layer** is three tensors:

| tensor | dtype | shape | content |
|---|---|---|---|
| `{name}.weight` | `U32` | `(out, in * bits / 32)` | the codes, `32 / bits` per word |
| `{name}.scales` | `BF16`, `F16`, or `F32` | `(out, in / group_size)` | one scale per group |
| `{name}.biases` | the same | `(out, in / group_size)` | one bias per group |

**Codes.** Row $i$ of `{name}.weight` holds the `in` codes of output $i$, packed
lowest bits first: with $p = 32 / \text{bits}$ codes per word, code $j$ of the row is

$$
c_{i,j} = \Big\lfloor \frac{\text{word}_{i,\,\lfloor j / p \rfloor}}{2^{\,\text{bits} \cdot (j \bmod p)}} \Big\rfloor \bmod 2^{\text{bits}},
$$

an unsigned integer in $[0, 2^{\text{bits}} - 1]$. A word is stored little-endian, so
at 8 bits the bytes of the tensor, in file order, are the codes in order.

**Scales and biases** are converted to float32 before use. A `BF16` value is the
upper 16 bits of a float32: put its 16-bit pattern in the high half of a 32-bit word
with zeros below. The kit's `bf16_to_f32(patterns)` does this for a `uint16` array;
you may call it. `F16` is IEEE half precision.

**Weights.** Group $g$ of a row covers the inputs $g \cdot \texttt{group\_size} \le j <
(g + 1) \cdot \texttt{group\_size}$, and

$$
w_{i,j} = c_{i,j} \cdot \text{scales}_{i,\,\lfloor j / \texttt{group\_size} \rfloor} + \text{biases}_{i,\,\lfloor j / \texttt{group\_size} \rfloor},
$$

computed in float32. A scale may be negative.

**Errors.**

- `KeyError` if any of the three tensors is not in the file. In particular a tensor
  that is not quantized, such as a norm gain `model.norm.weight`, has a `.weight` but
  no scales: `load_quantized(buf, "model.norm", ...)` raises `KeyError`.
- `ValueError` if `{name}.weight` is not `U32`; if `in`, which is the number of
  words per row times `32 / bits`, is not a multiple of `group_size`; if scales or
  biases do not have the shape `(out, in / group_size)`; or if their dtype is not one
  of the three. These are what a wrong `bits` or `group_size` looks like.

**Cost.** The bytes you request from `buf` (the sum of `b - a` over all slices) may
not exceed the bytes of the layer's three tensors plus **three times the header**
($3 \cdot (8 + n)$). Parse the header, then slice exactly the three tensors.
`buf[:]`, `bytes(buf)`-style copies, and `buf.read()` are charged for everything they
return.

Read the file and dequantize yourself. Not allowed: the `safetensors` package, the
kit's `SafetensorsFile`, `RealModel`, `dequantize`, and `unpack_codes`, and MLX's
`mx.load` and `mx.dequantize`.

## Example

A complete file of 264 bytes with one layer, `proj`, of 1 output and 8 inputs, at
8 bits with groups of 4:

```
offset  bytes
     0  f0 00 00 00 00 00 00 00                          n = 240
     8  {"__metadata__":{"format":"mlx"},
        "proj.weight":{"dtype":"U32","shape":[1,2],"data_offsets":[0,8]},
        "proj.scales":{"dtype":"BF16","shape":[1,2],"data_offsets":[8,12]},
        "proj.biases":{"dtype":"BF16","shape":[1,2],"data_offsets":[12,16]}}
        (233 bytes of JSON on one line, then 7 spaces)
   248  00 01 02 03  05 0a 80 ff                         proj.weight: two words
   256  00 3f  80 be                                     proj.scales
   260  80 bf  00 40                                     proj.biases
```

```python
load_quantized(buf, "proj", bits=8, group_size=4)
# array([[-1.0, -0.5, 0.0, 0.5, 0.75, -0.5, -30.0, -61.75]], dtype=float32)
```

The two words are `0x03020100` and `0xFF800A05`; their codes, lowest byte first, are
`0, 1, 2, 3` and `5, 10, 128, 255`. The scales are the bfloat16 patterns `0x3F00` and
`0xBE80`, that is $0.5$ and $-0.25$; the biases `0xBF80` and `0x4000` are $-1$ and
$2$. The first group is $c \cdot 0.5 - 1$, the second $c \cdot (-0.25) + 2$: code 128
gives $-30$ and code 255 gives $-61.75$. Read as signed bytes, those two codes would
be $-128$ and $-1$.

At 4 bits the same eight codes would fit in one word: for the codes `0, 1, 2, 3, 5,
10, 8, 15` it is `0xF8A53210`, the codes from right to left in hexadecimal.

In the real file, `model.layers.0.self_attn.k_proj.weight` has shape `[512, 640]`:
512 outputs, $640 \cdot 4 = 2560$ inputs, and 40 groups of 64 per row. Its three
tensors are 1,392,640 bytes, at three places spread over 277 MB of the file; the
header is 105,448 bytes.

## Constraints

- The result is compared with what MLX computes (`mx.dequantize` on float32 scales
  and biases) to a relative tolerance of `1e-6` and an absolute tolerance of `1e-7`,
  and must be a float32 array.
- The cases that need no download use files assembled in the test and two small
  checkpoints written by MLX (`mx.save_safetensors`): 8-bit codes in groups of 64
  with bfloat16 scales, and 4-bit codes in groups of 32 with float16 and float32
  scales. The byte limit applies to them as well.
- The real file is `model.safetensors` of `lmstudio-community/granite-4.2-3b-MLX-8bit`:
  3,888,771,048 bytes, 927 tensors. Three layers of shape `(512, 2560)` are read,
  with a limit of 1.709 MB each. Half of the file lies beyond byte $2^{31}$, so
  offsets do not fit in 32-bit integers.
- `in` is at most 2560 here; a layer fits in memory many times over. The time limit
  per case is 10 seconds, and a correct solution needs a few milliseconds.

## Why this matters

This is the file format behind every "MLX 4-bit" and "MLX 8-bit" model on the Hub,
and a loader is this function in a loop. Getting it right by hand is what lets you
inspect a checkpoint without a framework: check that a conversion really is 8-bit,
compare one layer of two quantizations, or compute statistics over a 40 GB model on a
laptop, because a memory map plus a header lets you touch 1.4 MB and leave the rest
on disk. The mistakes are the quiet kind. Codes read as signed bytes, words read
big-endian, or bfloat16 read as float16 all produce a matrix of the right shape with
plausible magnitudes, and a model that loads, runs, and talks nonsense. And the
loader that works on every test file and fails on the real one is the one that kept a
file offset in 32 bits.
