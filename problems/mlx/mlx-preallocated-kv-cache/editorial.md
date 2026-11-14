## The idea

In PyTorch a preallocated cache exists to avoid copying: `buffer[..., a:b, :] = new`
writes into memory that is already there. MLX arrays do not work that way. An MLX
array is a node in a graph, and

```python
self.keys[..., a:b, :] = keys
```

is shorthand for "let `self.keys` name a new array: the old one with this slice
replaced". When it is evaluated, MLX allocates a buffer of the full capacity, copies
the old contents, and writes the slice. Measured on this machine (M1 Max, MLX
0.32.4): 200 single-slot writes into a 16 MiB buffer take 0.35 ms each, the cost of
copying 16 MiB, and raise peak memory by two more buffers of that size. So the
step-allocated cache copies its buffer on every token, exactly as concatenation
does. It is not faster.

What differs is the *size* of the buffers that are allocated and freed. MLX does not
return a freed buffer to the system; it keeps it in a cache (`mx.get_cache_memory`)
and hands it to the next allocation that it fits. A cache whose capacity stays at 256
for 256 tokens asks, at every token, for a buffer of the size it has just freed: a
few buffers circulate and nothing new is allocated. A cache that grows by one slot
per token asks for a size nobody has freed yet, every time. Each old buffer goes into
the buffer cache and stays there.

Measured here, one token per update, each update followed by `mx.eval` of the fetched
arrays, memory relative to the start:

| cache | tokens | in use | peak | kept by the allocator afterwards |
|---|---|---|---|---|
| steps of 256, 1 KiB per token and buffer | 600 | 1.5 MiB | 4.5 MiB | 7.5 MiB |
| concatenation, same | 600 | 1.2 MiB | 3.6 MiB | 68.5 MiB |
| steps of 256, 4 KiB per token and buffer | 1,024 | 8 MiB | 24 MiB | 52 MiB |
| concatenation, same | 1,024 | 8 MiB | 24 MiB | 2.9 GiB |

Memory in use and peak memory are the same for both designs (peak is three times the
cache because of the copy). The last column is the difference, and it grows
quadratically with the length of the generation: the sum of every buffer size the
concatenating cache ever had. MLX stops keeping buffers only at its cache limit,
which defaults to most of the machine's memory (60.8 GiB on this 64 GB machine).

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

Growth drops the unused tail (`[..., :previous, :]`) before adding whole steps, so a
long prompt followed by single tokens wastes at most `step - 1` slots, however the
prompt length relates to the step. The returned slices are what keeps the tail out of
attention: an unused slot holds zeros, a zero key scores $0$, not $-\infty$, and a
softmax would give it weight (the subject of `preallocated-kv-cache`).

Everything here is lazy. The `mx.zeros`, the concatenation, and the slice updates
become part of the graph of the forward pass, and the model's one evaluation per token
(of the logits, which depend on the fetched slices, which depend on the buffers)
computes them. A generation loop should still pass the cache state to `mx.eval`, as
in `mlx-lazy-eval-placement`.

## Common mistakes

- **Concatenating on every update.** Correct contents, a new shape per token, and the
  allocator behavior in the table: 70 MiB held where 9 would do, in the test.
- **Growing by exactly `T`.** The preallocated code path with none of its benefit:
  the capacity equals the offset and every update reallocates.
- **Returning the whole buffers.** After the first single-token update the caller
  gets 256 "tokens", 255 of them zeros that attention will attend to.
- **`mx.zeros(shape)` without a dtype.** The buffer is float32. Writing float16 keys
  into it casts them, the fetched arrays are float32, and from then on the model
  computes attention in float32 with a cache twice the size (see
  `mlx-dtype-discipline`).
- **Allocating one step whatever `T` is.** A 300-token prompt does not fit in 256
  slots; the slice assignment fails.
- **Forgetting the old contents when growing.** The first 256 tokens turn into zeros
  at token 257: generation derails exactly at a multiple of the step, a bug that
  short tests never see.
- **Appending the new room to the whole old buffer** instead of to its used part.
  After a 300-token prompt (capacity 512) and a second chunk of 300 tokens, the
  capacity is 1,024 instead of 812, and the unused slots are carried along by every
  later copy.
- **Sizing the value buffer with the key's head dimension.** Keys and values have the
  same width in most models, not in all.
- **`offset += 1`.** Right for decoding, wrong for the prompt, which arrives as one
  chunk of many tokens.
- **Evaluating inside `update_and_fetch`.** One evaluation per layer per token, each a
  round trip to the GPU, in the middle of a forward pass that would have been one
  graph.

## Complexity

Appending $n$ tokens one at a time changes the buffers' shape
$\lceil n / \text{step} \rceil$ times. Every update copies both buffers once when it
is evaluated, $O(C)$, so generation is quadratic in the context length with either
design; the constant is one `memcpy` per buffer per token. Memory held by the
allocator stays within a small multiple of the cache size.

## In the wild

This is `mlx_lm.models.cache.KVCache`, with `step = 256` as a class attribute; its
`trim(n)` simply lowers `offset`, which works because the fetch never looks past it.
`mlx_lm.generate` adds the other half of the remedy: it calls `mx.clear_cache()`
every 256 generated tokens, right when the cache has outgrown its buffers and the old
sizes have become useless. When an MLX program's memory climbs while its arrays do
not, compare `mx.get_active_memory()` with `mx.get_cache_memory()`: if the second one
is growing, some array is changing shape in a loop.
