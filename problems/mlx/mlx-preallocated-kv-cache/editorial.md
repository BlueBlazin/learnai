## The idea

In PyTorch, a preallocated KV cache avoids copying because `buffer[..., a:b, :] = new` mutates memory in place. MLX arrays work differently. Every MLX array is an immutable node in a computation graph, so

```python
self.keys[..., a:b, :] = keys
```

binds `self.keys` to a new array representing the old buffer with that slice replaced. When evaluated, MLX allocates a full-capacity buffer, copies the old contents, and writes the new slice. Measured on an M1 Max (MLX 0.32.4), 200 single-slot writes into a 16 MiB buffer take 0.35 ms each (the cost of copying 16 MiB) and increase peak memory by two extra buffers of that size. Because it copies on every token, a step-allocated cache is no faster than concatenation.

The real difference is the size of the buffers being allocated and freed. When MLX frees a buffer, it keeps the memory in an internal pool (`mx.get_cache_memory`) and hands it to the next allocation of a matching size. While capacity stays fixed at 256 for 256 tokens, every token requests the exact buffer size that was just freed. A few buffers circulate and no new system memory is claimed. By contrast, a cache that grows by one slot per token requests a never-before-seen size on every step, so every freed buffer sits unused in the pool.

Measured on one token per update with `mx.eval` called on the fetched arrays after each step (relative to starting memory):

| cache | tokens | in use | peak | kept by the allocator afterwards |
|---|---|---|---|---|
| steps of 256, 1 KiB per token and buffer | 600 | 1.5 MiB | 4.5 MiB | 7.5 MiB |
| concatenation, same | 600 | 1.2 MiB | 3.6 MiB | 68.5 MiB |
| steps of 256, 4 KiB per token and buffer | 1,024 | 8 MiB | 24 MiB | 52 MiB |
| concatenation, same | 1,024 | 8 MiB | 24 MiB | 2.9 GiB |

Active memory and peak memory are comparable for both designs (peak is three times the cache size because of the copy). The final column diverges dramatically and grows quadratically with sequence length for concatenation, summing every distinct buffer size ever allocated. MLX stops caching freed buffers only when it hits its cache limit, which defaults to most of the machine's RAM (60.8 GiB on a 64 GB system).

## Reference solution (MLX)

```python
class KVCache:
    def __init__(self, step=256):
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        previous = self.offset
        new = keys.shape[2]
        if self.keys is None or previous + new > self.keys.shape[2]:
            batch, heads, _, key_dim = keys.shape
            room = -(-new // self.step) * self.step
            more_keys = mx.zeros((batch, heads, room, key_dim), keys.dtype)
            more_values = mx.zeros((batch, heads, room, values.shape[3]), values.dtype)
            if self.keys is None:
                self.keys, self.values = more_keys, more_values
            else:
                self.keys = mx.concatenate([self.keys[..., :previous, :], more_keys], axis=2)
                self.values = mx.concatenate([self.values[..., :previous, :], more_values], axis=2)

        self.offset = previous + new
        self.keys[..., previous : self.offset, :] = keys
        self.values[..., previous : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
```

When growing, the cache trims the unused tail (`[..., :previous, :]`) before concatenating the new steps. That guarantees a long prompt followed by single-token decoding wastes at most `step - 1` slots regardless of the prompt length. Returning only `[:self.offset]` keeps zero-filled padding out of attention. A zero key produces an attention logit of $0$ rather than $-\infty$, so softmax would assign non-zero weight to unused slots (as covered in `preallocated-kv-cache`).

Every operation here is lazy. `mx.zeros`, `mx.concatenate`, and the slice assignments simply extend the forward pass graph. When the model evaluates the logits at the end of the step, the buffers are materialized as part of that graph. A full generation loop should still include the cache state in its `mx.eval` call (as covered in `mlx-lazy-eval-placement`).

## Common mistakes

- **Concatenating on every update.** This produces the right values, but changes buffer shape on every token and balloons allocator memory (70 MiB instead of 9 MiB in the memory test).
- **Growing by exactly `T`.** This keeps capacity equal to `offset`, so every update reallocates and gives up the entire benefit of step allocation.
- **Returning the full buffers.** After the first single-token update, attention receives 256 tokens, 255 of which are zeros that distort softmax weights.
- **Calling `mx.zeros(shape)` without a dtype.** `mx.zeros` defaults to float32. Writing float16 keys into a float32 buffer upcasts them, causing attention to run in float32 with double the memory footprint (see `mlx-dtype-discipline`).
- **Allocating only one `step` regardless of `T`.** A 300-token prompt overflows a 256-slot buffer and crashes the slice assignment.
- **Dropping previous contents on growth.** If growth allocates a fresh buffer without copying `[:previous]`, the first 256 tokens become zeros on token 257. Generation silently derails at the first step boundary, which short unit tests miss.
- **Appending new room to the entire old buffer instead of the used prefix.** After a 300-token prompt (capacity 512) and another 300-token chunk, capacity becomes 1,024 instead of 812, carrying dead padding through every subsequent copy.
- **Using `keys.shape[3]` for the value buffer.** Keys and values have different head dimensions (`D_k != D_v`) in some architectures.
- **Incrementing `offset += 1`.** This works for single-token decoding but fails on multi-token prompt prefill.
- **Calling `mx.eval` inside `update_and_fetch`.** This forces a GPU synchronization per layer per token instead of evaluating one fused forward-pass graph.

## Complexity

Appending $n$ tokens one at a time changes the buffer shape $\lceil n / \text{step} \rceil$ times. Each evaluated update copies both buffers once in $O(C)$ time, where $C$ is the current buffer capacity, so total generation work is quadratic in context length for both designs with a constant factor of one copy per buffer per token. Total memory held by the allocator stays bounded by a small constant multiple of the live cache size.

## In the wild

This implementation matches `mlx_lm.models.cache.KVCache`, which sets `step = 256` as a class attribute. Its `trim(n)` method simply subtracts `n` from `offset` because `update_and_fetch` never reads past `offset`.

`mlx_lm.generate` pairs this with a periodic `mx.clear_cache()` every 256 generated tokens, right after the cache grows and older buffer sizes are no longer needed. When debugging an MLX program whose memory climbs even though live tensors stay small, compare `mx.get_active_memory()` and `mx.get_cache_memory()`. If cached memory is growing steadily, an array is changing shape inside a loop.
