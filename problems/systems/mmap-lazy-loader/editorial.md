## The idea

A safetensors file consists of a JSON table of contents followed by raw tensor bytes. Memory-mapping that file turns a byte range `a:b` into a virtual address range without reading from disk—the operating system faults pages in only when code touches them. A lazy loader built on these two properties has three distinct phases, each touching only what it needs:

1. **Open.** Read the 8-byte header length and the $n$ bytes of JSON. From that point on, the loader knows every tensor's name, dtype, shape, and byte range, which is enough to answer `keys()`, plan a device map, or count parameters without reading any tensor data.
2. **First use of a tensor.** Slice only that tensor's byte range and wrap it with `np.frombuffer`. This produces an array backed directly by the mapped pages, with no copy and no reads outside the tensor.
3. **Later uses.** Return the cached array from step 2. Storing materialized arrays in a dict keyed by name ensures lazy loading does not turn into repeated reads.

Two cases cannot be returned directly as a view.

**bfloat16.** NumPy has no `bfloat16` dtype, so mapped bytes cannot be viewed directly as floating-point numbers. They must be decoded into new memory. Because a `bfloat16` value is a `float32` value with its lower 16 mantissa bits truncated, decoding is an exact widening and left shift for every bit pattern, including infinities and NaN:

$$
\text{float32 bits} = \text{bf16 bits} \ll 16 .
$$

**Unaligned tensors.** A `float32` array is aligned when its memory address is a multiple of 4. Because the memory map starts at a page boundary, a tensor's address is aligned if and only if its file offset $8 + n + \texttt{begin}$ is a multiple of its element size. Writers usually pad the header so the data section starts on an 8-byte boundary, but the format does not require padding between tensors. A 3-byte mask followed by a float matrix places the matrix at an odd offset. NumPy still creates a view there with `flags.aligned == False`. Arithmetic works on x86 and ARM at a speed penalty, but downstream C extensions, vectorized kernels, or GPU upload paths that assume aligned pointers can fail. Copying only the unaligned tensors into NumPy-allocated memory fixes alignment without paying for copies on aligned tensors.

`load_subset` adds one more rule: validate every requested name before reading any tensor. On a large checkpoint, the first tensors in a subset can be gigabytes, and a typo in the last name should not waste minutes of I/O before raising `KeyError`.

## Reference solution (Python)

```python
class LazyCheckpoint:
    def __init__(self, file):
        self._file = file
        n = int.from_bytes(file[0:8], "little")
        entries = json.loads(bytes(file[8 : 8 + n]))
        entries.pop("__metadata__", None)
        self._entries = entries
        self._data_start = 8 + n
        self._arrays = {}

    def keys(self):
        return list(self._entries)

    def get(self, name):
        if name not in self._arrays:
            self._arrays[name] = self._materialize(self._entries[name])
        return self._arrays[name]

    def load_subset(self, names):
        names = list(names)
        for name in names:
            if name not in self._entries:
                raise KeyError(name)
        return {name: self.get(name) for name in names}

    def _materialize(self, info):
        begin, end = info["data_offsets"]
        start = self._data_start + begin
        raw = self._file[start : self._data_start + end]
        dtype = np.dtype(DTYPES[info["dtype"]])
        flat = np.frombuffer(raw, dtype=dtype)
        if info["dtype"] == "BF16":
            flat = (flat.astype(np.uint32) << 16).view(np.float32)
        elif start % dtype.itemsize:
            flat = flat.copy()
        return flat.reshape(info["shape"])
```

Converting `names = list(names)` is essential because `names` may be a single-pass generator and must be iterated twice. The dict comprehension preserves the first occurrence of any repeated name, and subsequent `get` calls hit the cache.

## Common mistakes

- **Reading the whole file to parse the header.** Calling `file.read()` or `file[:]` before slicing produces the right arrays but charges the entire file upfront. Even with a memory map, this marks the whole file range as read, and with a standard file handle it allocates the full checkpoint size in RAM.
- **Loading eagerly in `__init__`.** Materializing every tensor during construction defeats lazy loading.
- **Skipping the cache.** Re-slicing on every `get` passes value equality checks, but re-decodes every `BF16` tensor, re-copies every unaligned tensor, and breaks object identity across callers.
- **Copying every tensor.** Calling `np.array(view)` or `bytes(file[a:b])` doubles memory usage for tensors that could have been zero-copy views.
- **Using the wrong offset origin.** `data_offsets` are relative to the start of the data section at byte $8 + n$. Measuring from byte 8 returns correctly shaped arrays filled with header bytes.
- **Returning `BF16` as integers or `float16`.** Returning raw `uint16` values pushes decoding onto the caller, while viewing them as `float16` silently produces finite, plausible, wrong numbers because `float16` and `bfloat16` allocate exponent and mantissa bits differently.
- **Skipping the alignment check.** Trusting every view passes on checkpoints from writers that pad every tensor, then fails on any file with unaligned offsets.
- **Validating lazily in `load_subset`.** Calling `get` inside a loop raises `KeyError` only after reading all valid tensors that appeared earlier in `names`.

## Complexity

Opening takes $O(n)$ time for an $n$-byte header. The first `get` call takes $O(1)$ time for an aligned non-`BF16` view (the OS faults pages in when the caller reads elements) and time proportional to the tensor's byte size when decoding `BF16` or copying an unaligned tensor. Subsequent `get` calls are $O(1)$ dict lookups. Memory beyond the memory map consists of the parsed header, the cache dict, and any decoded or copied tensors.

## In the wild

- The safetensors format was built for lazy zero-copy loading. Its README highlights inspecting a checkpoint without loading it and loading tensor subsets as primary goals, noting that the header is the only part that requires copying.
- `safe_open(path)` in the `safetensors` library exposes this same interface (`keys()` and `get_tensor(name)`), reading each tensor only when requested.
- MLX and llama.cpp (GGUF) both memory-map weight files, which allows multi-gigabyte models to appear loaded in under a second when pages are already in the OS page cache.
- Most modern checkpoints store weights in `bfloat16`, so the decode-to-`float32` path is the common case rather than an edge case.
