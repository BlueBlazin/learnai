## The idea

Memory on an inference device is spent in a fixed order, and the calculation follows
it.

1. **Set the reserve aside.** A forward pass needs working memory (activations, the
   logits of a batch, temporary buffers), the framework holds memory of its own, and
   an allocator never packs perfectly. None of that is weights or cache, and none of it
   can be computed from the model's configuration, so engines take it as a share of
   the device: vLLM's `--gpu-memory-utilization 0.9` is a 10 percent reserve.
2. **Load the weights.** They are a fixed cost, paid once whatever the batch.
3. **Give the rest to the KV cache.** This is the only part that scales with traffic.

$$
\text{cache\_bytes} = \text{memory} - \left\lceil \tfrac{\text{memory} \cdot p}{100} \right\rceil - \text{weights}
$$

A paged cache hands that memory out in blocks of `block_tokens` tokens, so the unit of
account is the block, not the byte: the pool has $N = \lfloor \text{cache\_bytes} /
\text{block\_bytes} \rfloor$ blocks, a sequence of $c$ tokens holds $\lceil c / B
\rceil$ of them, and

$$
\text{sequences} = \left\lfloor \frac{N}{\lceil c / B \rceil} \right\rfloor, \qquad
\text{context} = \left\lfloor \frac{N}{\text{batch}} \right\rfloor \cdot B .
$$

**Every rounding goes the safe way.** A capacity is a promise, and each step rounds in
the direction that keeps it: the reserve up (set aside at least the share), the pool
down (a partial block cannot hold a token), the blocks of a sequence up (a partly
filled block is still taken), the final quotient down (a sequence that half fits does
not fit). Flip any one and there is an input for which the answer is one too large.

**Why the order matters.** The reserve is a share of the device. Applying the
percentage later, to the number of sequences, scales the weights too: it pretends that
10 percent of the weights can be left out. On the example that is
$\lfloor 0.9 \cdot \lfloor (M - W) / \text{block} \rfloor \rfloor = 12{,}668$ blocks
instead of 12,110, which admits two more 4096-token sequences than the device holds.

**Integers.** Byte counts here pass $2^{34}$ and their products with a percentage
pass $2^{40}$. A float64 still represents those exactly, but `memory * 0.9` is not
`memory * 9 / 10`: `0.9` is not a binary fraction, so the product is off by a fraction
of a byte in one direction or the other, and the floor that follows can land on the
wrong side of a block. Integer arithmetic has no such cases to think about.

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

The two functions are the same inequality solved for different unknowns:
`batch * ceil(context / B) <= N`.

## Common mistakes

- **Overhead applied after the division.** Scaling the block count (or the sequence
  count) by `1 - p` instead of reserving a share of the device first. It
  underestimates the reserve by `p` percent of the weights, which for a model that
  fills most of the device is most of the reserve.
- **Overhead taken from the remainder.** Reserving `p` percent of `memory - weights`
  is the same mistake in another order. The working memory of a forward pass does not
  get smaller because the weights are large.
- **Forgetting the weights.** The KV cache per token is so much smaller than the
  model that a budget without the weights looks plausible: 69 sequences instead of 47
  on the example.
- **Rounding up at the end.** `ceil(N / blocks_per_sequence)` counts the sequence that
  only partly fits. The full-context row of the example shows it: 12,110 blocks hold
  one 8192-block sequence, not two.
- **A negative budget.** When the weights exceed what the reserve leaves, the
  remainder is negative, and `//` rounds it toward minus infinity: `-5 // 3` is `-2`.
  Clamp the pool at zero.
- **Rounding the reserve down.** The difference is at most one byte, and one byte is
  enough: when the budget is exactly a whole number of blocks with the truncated
  reserve, the correct reserve leaves one block fewer.
- **Bytes instead of blocks.** Dividing the cache bytes by `context * token_bytes`
  ignores that a sequence of 1000 tokens holds 63 blocks of 16, which is 1008 tokens'
  worth, and that `max_context` cannot be 24,221 tokens when blocks hold 16.
- **Dropping the partial block of a sequence.** `context // block_tokens` charges a
  1000-token sequence 62 blocks; its last 8 tokens need a 63rd.

## Complexity

$O(1)$: a handful of integer operations.

## In the wild

- **vLLM** sizes its cache at startup this way: the requested fraction of GPU memory
  (`--gpu-memory-utilization`), minus the memory the model occupies after loading,
  becomes KV cache blocks of 16 tokens, and it reports the resulting maximum
  concurrency for sequences of the model's full length.
- **Real pools are a little smaller** than this model: block tables and per-sequence
  metadata take memory, and prefix sharing lets several sequences hold one block
  (`paged-kv-allocator`), which is the one case where the count can exceed this one.
- **On a laptop** the device is unified memory shared with the operating system, so
  the reserve is larger and softer; `max_context` with `batch = 1` is the "how long a
  document can I load" number, and it is why the 8-bit build of a model matters more
  for context length than for speed: on 8 GiB the BF16 weights of the example leave
  5,008 tokens of cache and the 8-bit weights 46,896.
