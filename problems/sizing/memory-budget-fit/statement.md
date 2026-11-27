Implement `max_sequences` and `max_context` to compute how many fixed-length sequences fit in device memory and the maximum context length a fixed batch of sequences can hold. Both functions reserve a fraction of device memory, subtract the model weights, and divide the remaining capacity into fixed-size KV cache blocks.

## Specification

```python
def max_sequences(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, context: int, overhead_percent: int
) -> int: ...
def max_context(
    memory_bytes: int, weight_bytes: int, cfg: dict, dtype: str, batch: int, overhead_percent: int
) -> int: ...
```

**Model configuration and cache element size.** `cfg` is a dict describing the KV cache geometry. Read these four keys and ignore any others:

| key | meaning |
|---|---|
| `n_layers` | number of transformer layers $L$ |
| `n_kv_heads` | number of key/value heads $H_{kv}$ |
| `head_dim` | width of one head $h$ |
| `block_tokens` | tokens stored per cache block |

`dtype` names the element type of the KV cache and determines `itemsize`: `"float32"` (4 bytes), `"float16"` (2 bytes), `"bfloat16"` (2 bytes), or `"int8"` (1 byte).

**The memory model.** A device with `memory_bytes` bytes is partitioned in three steps.

1. **Reserve.** Set aside `overhead_percent` percent of the *whole device* for forward-pass activations, framework allocations, and fragmentation. Round the reserve **up** to a whole byte:

$$
\text{reserved} = \left\lceil \frac{\text{memory\_bytes} \cdot \text{overhead\_percent}}{100} \right\rceil .
$$

2. **Weights.** Subtract `weight_bytes`, which is fixed regardless of batch size.
3. **KV cache.** Assign whatever remains to the KV cache:

$$
\text{cache\_bytes} = \text{memory\_bytes} - \text{reserved} - \text{weight\_bytes}.
$$

**Blocks.** The KV cache is managed as a pool of fixed-size blocks, as in paged attention. Each block stores the keys and values for `block_tokens` tokens of one sequence across all $L$ layers:

$$
\text{block\_bytes} = \text{block\_tokens} \cdot 2 \cdot L \cdot H_{kv} \cdot h \cdot \text{itemsize}.
$$

The pool holds

$$
N = \max\!\left(0,\ \left\lfloor \frac{\text{cache\_bytes}}{\text{block\_bytes}} \right\rfloor\right)
$$

whole blocks. Partial blocks cannot be used. If the reserve and weights together exceed `memory_bytes`, the pool has `0` blocks.

A sequence of $c$ tokens occupies $\lceil c / \text{block\_tokens} \rceil$ whole blocks. Its last block counts in full even when partly filled, and blocks are not shared between sequences.

**`max_sequences(..., context, ...)`** returns the largest number of sequences that can each hold `context` tokens simultaneously:

$$
\left\lfloor \frac{N}{\lceil \text{context} / \text{block\_tokens} \rceil} \right\rfloor .
$$

**`max_context(..., batch, ...)`** returns the largest context length, in tokens, that `batch` sequences can all hold simultaneously. Every sequence receives the same number of whole blocks, so the result is a multiple of `block_tokens`:

$$
\left\lfloor \frac{N}{\text{batch}} \right\rfloor \cdot \text{block\_tokens}.
$$

Both functions return a non-negative Python `int`, where `0` means nothing fits. Perform all calculations using integer arithmetic without converting any intermediate value to a float. Do not modify any input arguments.

## Example

Granite 4.2 3B has 40 layers and 8 KV heads of width 64. One token costs $2 \cdot 40 \cdot 8 \cdot 64 \cdot 2 = 81{,}920$ bytes in BF16, so a block of 16 tokens costs 1,310,720 bytes. Its 3,659,737,600 parameters in BF16 occupy 7,319,475,200 bytes. On a 24 GiB device with a 10 percent reserve:

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
| sequences of 4096 | 256 blocks each, $\lfloor 12110 / 256 \rfloor$ | 47 |
| sequences of 1000 | $\lceil 1000 / 16 \rceil = 63$ blocks each, $\lfloor 12110 / 63 \rfloor$ | 192 |
| context for 8 | $\lfloor 12110 / 8 \rfloor = 1513$ blocks each, times 16 | 24,208 |

A single sequence at the model's full 131,072-token context needs 8,192 blocks, so one fits and two do not. On an 8 GiB device ($8 \cdot 2^{30}$ bytes), the same weights leave 313 blocks, giving `max_sequences(..., 4096, 10) == 1` and `max_context(..., 1, 10) == 5008`. On an 8 GB device ($8 \cdot 10^9$ bytes), the reserve and weights together exceed device memory, so both functions return `0`.

## Constraints

- `1 <= memory_bytes <= 10**13` and `0 <= weight_bytes <= 10**13`. Weights may exceed `memory_bytes`.
- `0 <= overhead_percent <= 99`, an `int`.
- `1 <= n_layers <= 200`, `1 <= n_kv_heads <= 256`, `1 <= head_dim <= 1024`, and `1 <= block_tokens <= 4096`.
- `1 <= context <= 2,000,000` and `1 <= batch <= 100,000`.
- `dtype` is always one of `"float32"`, `"float16"`, `"bfloat16"`, or `"int8"`.
- Results are compared exactly.

## Why this matters

Inference servers perform this calculation once at startup to set their concurrency limit. For example, vLLM takes a fraction of GPU memory (`--gpu-memory-utilization`, 0.9 by default), subtracts the loaded weights, and turns the remainder into 16-token KV cache blocks. Dividing the block pool by the blocks needed for one full-length sequence gives the maximum concurrency it can safely promise.

Every rounding step has a safe direction. Rounding the reserve down, counting a partial block, or forgetting that a sequence's final block is allocated in full can admit one more sequence than the hardware holds. That mistake surfaces mid-generation as an out-of-memory error or a preemption rather than a clean refusal at admission. Running the same calculation in reverse answers the local-inference question of how long a context fits in unified memory at a given quantization.
