## The idea

Memory on an inference device is allocated in a fixed order, and the sizing calculation follows that order.

1. **Set the reserve aside.** A forward pass needs working memory for activations, batch logits, and temporary buffers. The framework also holds memory of its own, and allocators leave fragmentation. Because this overhead cannot be derived from the model config alone, serving engines reserve a share of the whole device (for example, vLLM's `--gpu-memory-utilization 0.9` leaves a 10 percent reserve).
2. **Load the weights.** Model weights are a fixed cost paid once regardless of batch size.
3. **Give the rest to the KV cache.** This is the only portion that scales with concurrency and context length.

Writing $p$ for `overhead_percent`, the bytes left for the cache are

$$
\text{cache\_bytes} = \text{memory} - \left\lceil \frac{\text{memory} \cdot p}{100} \right\rceil - \text{weights}.
$$

A paged KV cache allocates memory in blocks of $B = \text{block\_tokens}$ tokens, so the unit of accounting is the block rather than the byte. The pool contains $N = \max(0, \lfloor \text{cache\_bytes} / \text{block\_bytes} \rfloor)$ whole blocks, and a sequence of $c$ tokens requires $\lceil c / B \rceil$ blocks. Both target quantities follow directly:

$$
\text{sequences} = \left\lfloor \frac{N}{\lceil c / B \rceil} \right\rfloor, \qquad
\text{context} = \left\lfloor \frac{N}{\text{batch}} \right\rfloor \cdot B .
$$

**Every rounding goes the safe way.** A capacity bound is a promise, so each step rounds conservatively. The reserve rounds up so at least the requested share is kept free. The pool count rounds down because a partial block cannot hold a sequence. Blocks per sequence round up because a partly filled block is still allocated in full. The final quotient rounds down because a sequence that half fits does not fit. Flipping any of these four directions creates inputs where the answer is one too high.

**Why the order matters.** The reserve is a share of the whole device. Applying the percentage later to the block or sequence count scales the weights as well, as if $p$ percent of the weights could be omitted. In the worked example, $\lfloor 0.9 \cdot \lfloor (M - W) / \text{block\_bytes} \rfloor \rfloor$ yields 12,668 blocks instead of 12,110, which admits two more 4,096-token sequences than the device can hold.

**Why integer arithmetic.** Byte counts here exceed $2^{34}$, and multiplying by a percentage exceeds $2^{40}$. Even though `float64` represents those integers exactly, `0.9` is not an exact binary fraction. Computing `memory * 0.9` can be off by a fraction of a byte, which can push a floor across a block boundary. Pure integer arithmetic avoids rounding drift completely.

## Reference solution (Python)

```python
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}


def pool_blocks(memory_bytes, weight_bytes, cfg, dtype, overhead_percent):
    reserved = (memory_bytes * overhead_percent + 99) // 100  # ceiling
    cache_bytes = memory_bytes - reserved - weight_bytes
    token_bytes = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    return max(0, cache_bytes // (cfg["block_tokens"] * token_bytes))


def max_sequences(memory_bytes, weight_bytes, cfg, dtype, context, overhead_percent):
    blocks = pool_blocks(memory_bytes, weight_bytes, cfg, dtype, overhead_percent)
    return blocks // -(-context // cfg["block_tokens"])


def max_context(memory_bytes, weight_bytes, cfg, dtype, batch, overhead_percent):
    blocks = pool_blocks(memory_bytes, weight_bytes, cfg, dtype, overhead_percent)
    return (blocks // batch) * cfg["block_tokens"]
```

Both functions solve the same inequality, `batch * ceil(context / B) <= N`, for different unknowns.

## Common mistakes

- **Applying overhead after the division.** Scaling the block count or sequence count by `1 - p / 100` underestimates the reserve by `p` percent of the weights. When weights fill most of the device, that omits most of the reserve.
- **Taking overhead from the remainder.** Reserving `p` percent of `memory - weights` makes the same mistake in a different order. Forward-pass working memory does not shrink when the weights are large.
- **Forgetting the weights.** Because KV cache per token is much smaller than the model, omitting `weight_bytes` still produces plausible-looking numbers (69 sequences instead of 47 on the example).
- **Rounding up at the end.** Using `ceil(N / blocks_per_sequence)` counts a sequence that only partly fits. On the full-context row of the example, 12,110 blocks hold one 8,192-block sequence, not two.
- **Not clamping a negative budget.** When weights exceed the unreserved memory, `cache_bytes` is negative and Python's `//` rounds toward $-\infty$ (for example, `-5 // 3 == -2`). Clamp the block pool at `0`.
- **Rounding the reserve down.** The error is at most one byte, which is enough to lose a block whenever the truncated reserve leaves an exact multiple of `block_bytes`.
- **Computing in bytes instead of blocks.** Dividing `cache_bytes` by `context * token_bytes` ignores block granularity. A 1,000-token sequence occupies 63 blocks of 16 (1,008 tokens' worth), and `max_context` must be a multiple of 16 rather than 24,221.
- **Dropping a sequence's partial block.** Using `context // block_tokens` charges a 1,000-token sequence 62 blocks, leaving no room for its final 8 tokens.

## Complexity

$O(1)$ time and space using a constant number of integer operations.

## In the wild

- **vLLM** sizes its KV cache at startup using this model. It takes the requested fraction of GPU memory (`--gpu-memory-utilization`), subtracts the memory used by the loaded weights, converts the remainder into 16-token blocks, and reports the resulting maximum concurrency for full-length sequences.
- **Real block pools are slightly smaller** because block tables and per-sequence metadata consume memory too. The one case where actual concurrency can exceed this bound is prefix sharing, where multiple sequences reference the same KV blocks (`paged-kv-allocator`).
- **Laptops with unified memory** share RAM with the operating system, so the reserve is larger and less rigid. Here `max_context` with `batch = 1` answers how long a document fits in memory, and explains why 8-bit quantization often matters more for context length than for speed. On 8 GiB, the BF16 weights from the example leave 5,008 tokens of cache, whereas 8-bit weights leave 46,896 tokens.
