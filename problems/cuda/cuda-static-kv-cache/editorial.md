## The idea

A growing cache carries its length in its shape. A static cache gives that up: the
buffers are `max_len` long from the first step, and the three things that used to
follow from the shape have to be said explicitly.

**Where to write.** Slots `pos` to `pos + T - 1`. Written as a slice,
`buffer[:, :, pos : pos + T] = k`, this needs the *value* of `pos` on the host,
because a slice is built from Python integers; with `pos` on a GPU the host waits
for the device at every step. `index_copy_(dim, index, source)` takes the slots as a
tensor: `pos + arange(T)` is computed on the device, and the write is queued like any
other kernel. `T` is not a problem: it is `k.shape[2]`, and shapes live on the host.

**What to attend to.** Slots `0` to `pos + i` for the `i`-th new token. The unused
tail cannot be sliced away without changing the shape, so it is masked: a bool row
per query, `True` up to and including the query's own slot. The model writes
$-\infty$ into the masked scores, the softmax turns them into exact zeros, and a
zero weight times a finite value contributes nothing. The comparison
`slots[None, :] <= (pos + arange(n_new))[:, None]` builds all rows at once, on the
device, in a shape that depends only on `n_new`.

**Which memory.** The same, always. `index_copy_` and `zero_` end in an underscore:
they write into the tensor they are called on and allocate nothing. Their
out-of-place twins return a new tensor; assigning that to `self.k[layer]` keeps the
numbers right and silently turns the static cache back into an allocating one. It
also breaks every consumer that captured the old tensor: a CUDA graph replays on the
addresses it recorded, and would keep reading the buffer that is no longer updated.

## Reference solution (PyTorch)

```python
class StaticCache:
    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, device, dtype):
        shape = (batch, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self._slots = torch.arange(max_len, device=device)

    def update(self, layer, k, v, pos):
        index = pos + self._slots[: k.shape[2]]
        self.k[layer].index_copy_(2, index, k)
        self.v[layer].index_copy_(2, index, v)
        return self.k[layer], self.v[layer]

    def mask(self, pos, n_new):
        last_visible = pos + self._slots[:n_new]
        return self._slots[None, :] <= last_visible[:, None]

    def reset(self):
        for buffer in (*self.k, *self.v):
            buffer.zero_()
```

`self._slots[:T]` is a view, and `pos + view` is a new `(T,)` tensor on the device:
a few bytes per step, served from the allocator's cache. Nothing in the class
depends on how many tokens are stored; the cache does not even keep a length. The
caller owns the position, as a device tensor it increments in place.

If the mask is right, `reset` is not needed for correctness, since the next request
masks whatever is left. It is specified because leaving one request's keys and
values in memory that the next request's code can reach is a bug waiting for an
off-by-one.

## Common mistakes

- **`torch.cat`.** Correct numbers; a new shape and two allocations per layer per
  step.
- **`index_copy` without the underscore, rebinding the buffer.** Correct numbers and
  shapes; a full-size allocation and copy per write, and a new `data_ptr()`.
- **`[torch.zeros(shape)] * n_layers`.** The list holds the same tensor `n_layers`
  times, and every layer overwrites the others' keys.
- **The mask with `<` instead of `<=`.** A token cannot see itself, and the first
  token of a request sees nothing: its softmax is over $-\infty$ only, which is NaN.
- **The mask one slot too wide.** The query attends to the first unused slot. With a
  zero tail the logits are slightly off at every step; with a reused buffer the
  model reads the previous request.
- **One mask row for a whole block.** Giving every token of a prefill the visibility
  of the block's last token lets tokens attend to their successors.
- **No dependence on `pos` at all** (everything visible).
- **`int(pos)` or `pos.item()`** to build a slice. One host synchronization per layer
  per step.
- **`pos` as a slice bound** (`buffer[:, :, pos:end].copy_(k)`). The same read,
  hidden in the indexing code.
- **`torch.arange(max_len)` without `device=`.** On a GPU it is a CPU tensor;
  `.to(device)` afterwards repairs the device and costs a transfer.
- **Buffers in the default dtype.** A float16 model with a float32 cache holds twice
  the memory and converts at every read.
- **Returning the filled prefix** (`buffer[:, :, :end]`). In-place writes, but a
  shape that changes with the step and a host read to compute `end`.
- **`reset` that allocates new buffers**, or that does nothing.

## Complexity

`update` copies $O(B \cdot H \cdot T \cdot D)$ elements and allocates a `(T,)` index;
`mask` is $O(\text{n\_new} \cdot \text{max\_len})$. Memory is fixed at
$2 \cdot L \cdot B \cdot H \cdot \text{max\_len} \cdot D$ elements from construction
on. The model's attention now runs over `max_len` slots at every step instead of
over the current length; that is the price of the static shape, and it is small
against a recompilation.

## In the wild

Hugging Face Transformers' `StaticCache` is this design: zero buffers of
`max_cache_len` slots per layer, `index_copy_` at a `cache_position` tensor in
`update`, in-place zeroing in `reset`, introduced so that `generate` can run under
`torch.compile`. The next two problems of this track build on it for exactly the
two consumers named above: a decode step compiled once, and a captured CUDA graph.

What was verified where: on the CPU build of torch 2.15, `index_copy_` writes in
place and keeps `data_ptr()`, an out-of-range index raises `IndexError`, and a
tensor used as a slice bound reads its value (visible as `Tensor.item()` failing on
the meta device). That the same read is a host synchronization on a GPU, and that
reserved memory stays flat over a decode loop, are taken from the documentation; the
GPU cases have not been run.
