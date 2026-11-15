Turn a memory budget into a concurrency limit: given a device, a model's weights, and
the geometry of its KV cache, compute how many sequences of a given length fit, and how
long a context a given number of sequences can hold. It is subtraction and two
divisions; the substance is the order of the steps and the direction of each rounding,
because every one of them has a plausible wrong choice that admits a request the
hardware cannot hold.

## Specification

```python
def max_sequences(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, context: int, overhead_percent: int
) -> int: ...
def max_context(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, batch: int, overhead_percent: int
) -> int: ...
```

**The memory model.** A device with `memory_bytes` bytes holds three things.

1. **A reserve.** `overhead_percent` percent of the *whole device* is kept free of
   weights and cache: it is where the activations of the forward pass, the framework's
   own allocations, and fragmentation go. The reserve is rounded **up** to a whole
   byte:

$$
\text{reserved} = \left\lceil \frac{\text{memory\_bytes} \cdot \text{overhead\_percent}}{100} \right\rceil .
$$

2. **The weights**: `weight_bytes` bytes, whatever the batch.
3. **The KV cache**, in what is left:

$$
\text{cache\_bytes} = \text{memory\_bytes} - \text{reserved} - \text{weight\_bytes}.
$$

**Blocks.** The cache is a pool of fixed-size blocks, as in a paged KV cache. One block
holds the keys and values of `block_tokens` tokens of one sequence in every layer:

$$
\text{block\_bytes} = \text{block\_tokens} \cdot 2 \cdot L \cdot H_{kv} \cdot h \cdot \text{itemsize},
$$

the per-token size of `kv-cache-bytes` times the block length. The pool has

$$
N = \max\!\left(0,\ \left\lfloor \frac{\text{cache\_bytes}}{\text{block\_bytes}} \right\rfloor\right)
$$

blocks: a partial block is no block, and when the reserve and the weights exceed the
device there is no pool. A sequence that holds $c$ tokens occupies
$\lceil c / \text{block\_tokens} \rceil$ whole blocks; its last block counts in full
even if it is partly filled. Blocks are not shared between sequences.

**`cfg`** is a dict. The functions use four of its keys and must ignore any others:

| key | meaning |
|---|---|
| `n_layers` | number of transformer blocks $L$ |
| `n_kv_heads` | number of key/value heads $H_{kv}$ |
| `head_dim` | width of one head $h$ |
| `block_tokens` | tokens per cache block |

**`dtype`** names the element type of the cache: `"float32"` (4 bytes per element),
`"float16"` (2), `"bfloat16"` (2), or `"int8"` (1).

**`max_sequences(..., context, ...)`** returns the largest number of sequences that
can each hold `context` tokens at the same time:

$$
\left\lfloor \frac{N}{\lceil \text{context} / \text{block\_tokens} \rceil} \right\rfloor .
$$

**`max_context(..., batch, ...)`** returns the largest context, in tokens, such that
`batch` sequences can all hold it at the same time. Every sequence gets the same
number of whole blocks, so the result is a multiple of `block_tokens`:

$$
\left\lfloor \frac{N}{\text{batch}} \right\rfloor \cdot \text{block\_tokens}.
$$

Both return a Python int that is never negative; `0` means "does not fit". All
arithmetic is on integers: no intermediate value may pass through a float. The
functions do not modify their arguments.

## Example

Granite 4.2 3B has 40 layers and 8 KV heads of width 64, so a token costs
$2 \cdot 40 \cdot 8 \cdot 64 \cdot 2 = 81{,}920$ bytes of BF16 cache and a block of 16
tokens costs 1,310,720 bytes. Its 3,659,737,600 parameters in BF16 occupy
7,319,475,200 bytes. Take a device with 24 GiB and a 10 percent reserve:

```python
cfg = {"n_layers": 40, "n_heads": 40, "n_kv_heads": 8, "head_dim": 64, "block_tokens": 16}
M = 24 * 2**30  # 25,769,803,776
W = 7_319_475_200

max_sequences(M, W, cfg, "bfloat16", 4096, 10)  # 47
max_sequences(M, W, cfg, "bfloat16", 1000, 10)  # 192
max_sequences(M, W, cfg, "bfloat16", 131072, 10)  # 1
max_context(M, W, cfg, "bfloat16", 1, 10)  # 193760
max_context(M, W, cfg, "bfloat16", 8, 10)  # 24208
max_context(M, W, cfg, "bfloat16", 64, 10)  # 3024
```

| step | arithmetic | value |
|---|---|---|
| reserved | $\lceil 25{,}769{,}803{,}776 \cdot 10 / 100 \rceil = \lceil 2{,}576{,}980{,}377.6 \rceil$ | 2,576,980,378 |
| cache bytes | $25{,}769{,}803{,}776 - 2{,}576{,}980{,}378 - 7{,}319{,}475{,}200$ | 15,873,348,198 |
| pool | $\lfloor 15{,}873{,}348{,}198 / 1{,}310{,}720 \rfloor$ | 12,110 blocks |
| sequences of 4096 | 256 blocks each: $\lfloor 12110 / 256 \rfloor$ | 47 |
| sequences of 1000 | $\lceil 1000 / 16 \rceil = 63$ blocks each: $\lfloor 12110 / 63 \rfloor$ | 192 |
| context for 8 | $\lfloor 12110 / 8 \rfloor = 1513$ blocks each, times 16 | 24,208 |

A single sequence at the model's full 131,072-token context needs 8192 blocks; one
fits, two do not. On a device with 8 GiB ($8 \cdot 2^{30}$ bytes) the same weights
leave 313 blocks: `max_sequences` at 4096 tokens is `1` and `max_context` for one
sequence is `5008`. On a device with 8 GB ($8 \cdot 10^9$ bytes) the reserve and the
weights add up to more than the device, and both functions return `0`.

## Constraints

- `1 <= memory_bytes <= 10**13` and `0 <= weight_bytes <= 10**13`; the weights may
  exceed the device.
- `0 <= overhead_percent <= 99`, an int.
- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= 256`, `1 <= head_dim <= 1024`,
  `1 <= block_tokens <= 4096`.
- `1 <= context <= 2,000,000` and `1 <= batch <= 100,000`.
- `dtype` is always one of the four names above.
- Results are compared exactly.

## Why this matters

This is the calculation an inference server makes once, at startup, and then lives
with: vLLM takes a fraction of the GPU (`--gpu-memory-utilization`, 0.9 by default),
subtracts what the loaded model occupies, and turns the remainder into KV cache blocks
of 16 tokens; the number of blocks divided by the blocks of one full-length sequence is
the concurrency it can promise. Every rounding in the chain has a safe direction. Round
the reserve down, count a partial block, or forget that a sequence's last block is
whole, and the plan admits one sequence more than fits; the failure then arrives in
the middle of a generation, as an out-of-memory error or a preemption, rather than as a
refusal at the door. The same arithmetic, run backwards, answers the laptop question:
with this much unified memory and this quantization, how long a context can I afford.
