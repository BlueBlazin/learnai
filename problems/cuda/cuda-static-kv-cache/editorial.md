## The idea

A dynamically growing cache encodes its current length in its tensor shape. A static cache fixes the buffer length at `max_len` from the first step, so three things must be handled explicitly.

**Where to write.** A block of `T` new tokens goes into slots `pos` through `pos + T - 1`. Writing with a slice such as `buffer[:, :, pos : pos + T] = k` requires Python integers as slice bounds, which forces a host read of `pos` and stalls the GPU at every step. Instead, `index_copy_(dim, index, source)` accepts a device tensor of slot indices. Computing `pos + arange(T)` stays entirely on the device, and the write is queued like any other kernel. Using `T = k.shape[2]` is free because tensor shapes already live on the host.

**What to attend to.** The $i$-th new token can attend to slots `0` through `pos + i`. Because slicing away the unused tail would change the tensor shape, we hide it with a boolean mask that is `True` up to and including each query's slot. Setting masked scores to $-\infty$ before softmax produces exact zero weights, so finite values in unused slots contribute nothing. With `slots = arange(max_len)`, broadcasting `slots[None, :] <= (pos + arange(n_new))[:, None]` builds all mask rows in one device operation with a shape that depends only on `n_new`.

**Which memory to use.** The buffers allocated in `__init__` must be reused for the lifetime of the cache. In-place methods with a trailing underscore (`index_copy_` and `zero_`) modify the existing tensor without allocating. Their out-of-place counterparts return a new tensor. Rebinding `self.k[layer]` to that new tensor still produces the right numbers, but it allocates on every step and breaks consumers like CUDA graphs that hold fixed pointers to the original buffer addresses.

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

`self._slots[:T]` is a zero-copy view, and `pos + view` creates a tiny `(T,)` tensor on the device from the caching allocator. The cache does not even store a current length. Instead, the caller owns `pos` as a device tensor and increments it in place.

With a correct mask, `reset` is not strictly required for numerical correctness because the next request masks out unvisited slots. Clearing the buffers is still good practice so an off-by-one bug cannot silently read keys and values from an earlier request.

## Common mistakes

- **Growing the cache with `torch.cat`.** The values match, but every step changes the shape and allocates two new tensors per layer.
- **Calling `index_copy` without the trailing underscore and rebinding the buffer.** Numbers and shapes look right, but each write allocates a full-size copy and changes `data_ptr()`.
- **Aliasing layers with `[torch.zeros(shape)] * n_layers`.** Every list entry points to the same tensor, so layers overwrite each other.
- **Using `<` instead of `<=` in the mask.** A token cannot attend to itself, and the first token sees no valid slots, causing softmax over all $-\infty$ values to produce `NaN`.
- **Making the mask one slot too wide.** Queries attend to the first unused slot. Against a zero tail this slightly skews logits, and against a reused buffer it reads stale data from the previous request.
- **Sharing one mask row across a multi-token block.** Giving every token in a prefill block the visibility of the block's last token lets tokens attend to future tokens.
- **Ignoring `pos` in the mask.** Making all slots visible attends to the entire unused buffer.
- **Calling `int(pos)` or `pos.item()` to build a slice.** This triggers a host-device synchronization on every layer and step.
- **Using `pos` as a slice bound (`buffer[:, :, pos:end].copy_(k)`).** Python slicing reads the tensor value onto the host under the hood.
- **Calling `torch.arange(max_len)` without `device=`.** On a GPU this creates a CPU tensor, and calling `.to(device)` later adds an unnecessary host-to-device transfer.
- **Leaving buffers in the default `float32` dtype.** A `float16` model with a `float32` cache doubles memory usage and converts dtypes on every read.
- **Returning only the filled prefix (`buffer[:, :, :end]`).** Writes happen in place, but the returned shape changes on every step and computing `end` requires a host read.
- **Allocating new buffers in `reset`, or leaving `reset` as a no-op.** Reallocating breaks fixed addresses, while doing nothing fails the zeroing requirement.

## Complexity

Let $L = \text{n\_layers}$, $B = \text{batch}$, $H = \text{n\_kv\_heads}$, and $D = \text{head\_dim}$. `update` copies $O(B \cdot H \cdot T \cdot D)$ elements and allocates a `(T,)` index tensor. `mask` runs in $O(\text{n\_new} \cdot \text{max\_len})$ time and memory. Total cache memory is fixed from construction at $2 \cdot L \cdot B \cdot H \cdot \text{max\_len} \cdot D$ elements. Attention now scores all `max_len` slots at every step rather than only the filled prefix, which is a small cost compared to recompilation or losing CUDA graph capture.

## In the wild

Hugging Face Transformers implements `StaticCache` with this exact pattern: zero-initialized buffers of length `max_cache_len` per layer, `index_copy_` using a `cache_position` tensor in `update`, and in-place zeroing in `reset` so `generate` can run under `torch.compile`. The next two problems in this track build directly on this cache for a compiled decode step and a captured CUDA graph.

Verification note: on the CPU build of PyTorch 2.15, `index_copy_` writes in place and preserves `data_ptr()`, an out-of-range index raises `IndexError`, and using a tensor as a slice bound reads its value (caught by `Tensor.item()` failing on the `meta` device). GPU host-synchronization behavior and flat reserved memory across decode steps follow the PyTorch documentation. The GPU cases have not been run on hardware.
