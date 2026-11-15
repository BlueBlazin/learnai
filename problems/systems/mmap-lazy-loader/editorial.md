## The idea

A safetensors file is a table of contents followed by raw tensors, and a memory map
turns "the bytes from `a` to `b`" into an address range without reading anything:
the operating system brings a page in when something touches it. A loader built on
those two facts has three moments, and each should touch only what it needs.

1. **Open.** Read the 8-byte length and the $n$ bytes of JSON. From then on the
   loader knows every tensor's name, dtype, shape, and position, which is enough to
   answer `keys()`, to plan a device map, or to count parameters. No tensor byte has
   been read.
2. **First use of a tensor.** Slice exactly its byte range and wrap it:
   `np.frombuffer` builds an array whose memory *is* the mapped range. Nothing is
   copied, and nothing around the tensor is touched.
3. **Later uses.** Return the array made in step 2. The cache is a dict from name
   to array; it is what makes "lazy" different from "re-read every time".

Two kinds of tensor cannot be handed out as a view.

**bfloat16.** numpy has no such dtype, so there is no way to describe the mapped
bytes as the numbers they are. They have to be decoded, and the result lives in new
memory. A bfloat16 value is a float32 with the low 16 bits of the mantissa removed,
so decoding is a widening and a shift, exact for every pattern, infinities and NaN
included:

$$
\text{float32 bits} = \text{bf16 bits} \ll 16 .
$$

**Unaligned tensors.** A `float32` array is aligned when its address is a multiple
of 4. The map begins at a page boundary, so the address of a tensor is aligned
exactly when its file offset $8 + n + \texttt{begin}$ is. Writers pad the header so
that the data section starts at a multiple of 8, but nothing in the format aligns
each tensor inside it: a 3-byte mask followed by a matrix of floats puts the matrix
at an odd offset. numpy will build a view there and mark it
`flags.aligned == False`. Arithmetic on it works on x86 and ARM, more slowly; code
that receives the pointer and assumes alignment (a C extension, a vectorized kernel,
a GPU upload path) may not. One copy into memory numpy allocates fixes it, and it is
taken only for the tensors that need it.

`load_subset` adds one rule: validate the request before paying for any of it. On a
real checkpoint the first names of a subset can be gigabytes, and a typo in the last
one should not cost a minute of I/O before the error.

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

`names = list(names)` matters: the argument may be a generator, and it is walked
twice. The dict comprehension keeps the first position of a repeated name, and the
second `get` of it is a cache hit.

## Common mistakes

- **Reading the file to find the header.** `file.read()` or `file[:]` followed by
  correct slicing gives correct arrays and has already paid for everything. On a
  map the bytes are not copied, but the loader has declared the whole range in use;
  with an ordinary file handle the same code allocates the size of the checkpoint.
- **Loading eagerly.** Building every array in the constructor "so that `get` is
  fast" is the loader this problem replaces.
- **No cache.** Slicing again on every `get` returns equal arrays, so a value test
  passes; but each BF16 tensor is decoded again and each unaligned one copied again,
  and two callers no longer share one array.
- **Copying everything.** `np.array(view)` or `bytes(file[a:b])` on the way doubles
  the memory of every tensor that was already usable where it lay.
- **Offsets from the wrong origin.** `data_offsets` count from the data section,
  byte $8 + n$. Counting from byte 8 returns arrays of the right shape filled with
  header text.
- **BF16 as integers, or as float16.** Returning the `uint16` patterns leaves the
  decoding to a caller who was promised weights; viewing them as `float16` gives
  finite, plausible, wrong numbers, because the two formats split their 16 bits
  differently.
- **Trusting every view.** Skipping the alignment test works for every checkpoint a
  mainstream writer produced and fails on the first file from somewhere else.
- **Validating as you go in `load_subset`.** Calling `get` in a loop raises the
  `KeyError` at the bad name, after the good ones before it have been read.

## Complexity

Opening costs $O(n)$ for a header of $n$ bytes. A first `get` costs the size of the
tensor only when it is decoded or copied, and $O(1)$ when it is a view; the pages
are read when the caller touches them. Every later `get` is a dict lookup. Memory
beyond the map is the header, the cache, and the decoded or copied tensors.

## In the wild

- The safetensors format was designed for this: its README lists inspecting a file
  without loading it and loading only some of its tensors among the goals, and
  describes the header as the only part that is not zero-copy.
- `safe_open(path)` of the `safetensors` package is the same interface: `keys()`
  and `get_tensor(name)`, each tensor read on request.
- MLX and llama.cpp (GGUF) both map their weight files, which is why a multi-gigabyte
  model appears to load in well under a second when the file is in the page cache.
- bfloat16 is what most current checkpoints are stored in, including this
  project's house models, so the "no numpy dtype" branch is the common path and not
  an exotic one.
