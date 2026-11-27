Compute the memory-bandwidth ceiling on transformer decode throughput and the batch size at which decoding stops being memory-bound. Generating a token reads every model weight from memory, and batching shares that weight read across sequences while still reading each sequence's KV cache separately.

## Specification

```python
def decode_tokens_per_s(weight_bytes: int, kv_bytes_read: int,
                        bandwidth_gb_per_s: float, batch: int) -> float: ...
def crossover_batch(weight_bytes: int, kv_bytes_read: int,
                    bandwidth_gb_per_s: float,
                    compute_tokens_per_s: float) -> int | None: ...
```

**Memory traffic model.** One decode step advances a batch of $b = \text{batch}$ sequences by one token each.

- Every model weight is read once per step regardless of batch size, because each weight matrix is streamed from memory and applied to all $b$ sequences' activations. That costs `weight_bytes` bytes.
- Each sequence's KV cache is read once per step, costing `kv_bytes_read` bytes **per sequence**. All sequences have the same cache size.
- No other memory traffic is counted. The total bytes read per step are

$$
\text{bytes}(b) = \text{weight\_bytes} + b \cdot \text{kv\_bytes\_read}.
$$

- Memory delivers at most `bandwidth_gb_per_s` gigabytes per second, where **1 GB is $10^9$ bytes** (the decimal unit used by hardware vendors). A step cannot finish faster than its bytes can be transferred.

**`decode_tokens_per_s(weight_bytes, kv_bytes_read, bandwidth_gb_per_s, batch)`.** Returns the upper bound on decode throughput in tokens per second, **summed over the batch**, as a `float`. Because each step emits $b$ tokens,

$$
M(b) = \frac{b \cdot \text{bandwidth\_gb\_per\_s} \cdot 10^9}{\text{weight\_bytes} + b \cdot \text{kv\_bytes\_read}} .
$$

The throughput of a single sequence is $M(b) / b$.

**`crossover_batch(weight_bytes, kv_bytes_read, bandwidth_gb_per_s, compute_tokens_per_s)`.** Here `compute_tokens_per_s` is the throughput $C$ (summed over the batch) that the processor's arithmetic units could sustain if memory were infinitely fast. In this model, $C$ does not depend on the batch size. Actual throughput is at most $\min(M(b), C)$. Decoding is *memory-bound* while $M(b) < C$ and *compute-bound* once $M(b) \ge C$.

This function returns the smallest integer batch $b \ge 1$ such that $M(b) \ge C$, as a Python `int`, or `None` if no batch satisfies the condition. Returning `None` is a valid outcome when `kv_bytes_read > 0`, because $M(b)$ increases with $b$ but stays strictly below $(\text{bandwidth\_gb\_per\_s} \cdot 10^9) / \text{kv\_bytes\_read}$.

## Example

The 8-bit MLX build of Granite 4.2 3B (`lmstudio-community/granite-4.2-3b-MLX-8bit`) is a single 3,888,771,048-byte file. Consider a machine with 400 GB/s of unified memory bandwidth, the figure Apple specifies for the M1 Max and M2 Max. At a context length of 4096 tokens, one sequence's KV cache in BF16 is $81920 \cdot 4096 = 335{,}544{,}320$ bytes.

```python
W = 3_888_771_048
K = 335_544_320

decode_tokens_per_s(W, 0, 400.0, 1)    # 102.86025972285525   weights only
decode_tokens_per_s(W, K, 400.0, 1)    # 94.68990005577633
decode_tokens_per_s(W, K, 400.0, 8)    # 486.8308002672813    60.9 per sequence

crossover_batch(W, K, 400.0, 800.0)    # 24
crossover_batch(W, 0, 400.0, 800.0)    # 8
crossover_batch(W, K, 400.0, 1200.0)   # None
crossover_batch(W, K, 400.0, 90.0)     # 1
```

A single stream cannot exceed $400 \cdot 10^9 / 3{,}888{,}771{,}048 = 102.9$ tokens per second even with an empty cache, and drops to 94.7 tokens per second when reading a 4096-token cache. A batch of 8 shares one pass over the weights, reaching $8 \cdot 400 \cdot 10^9 / (3{,}888{,}771{,}048 + 8 \cdot 335{,}544{,}320) = 486.8$ total tokens per second. That is 5.1 times the single-stream rate rather than 8 times, because each sequence still reads its own KV cache.

If the processor can compute 800 tokens per second, the memory-bandwidth bound reaches that rate at batch 24 ($M(23) = 792.7$, $M(24) = 803.9$). At 1200 tokens per second, it never catches up because $M(b)$ approaches $400 \cdot 10^9 / 335{,}544{,}320 = 1192.1$ from below.

## Constraints

- `1 <= weight_bytes <= 10**13`, `0 <= kv_bytes_read <= 10**12`, `0 < bandwidth_gb_per_s <= 10**5`, `1 <= batch <= 10**6`, and `0 < compute_tokens_per_s <= 10**9`.
- `decode_tokens_per_s` is compared against the reference with a relative tolerance of `1e-9`.
- `crossover_batch` is compared exactly. Test inputs avoid boundary cases where rounding could flip the answer. Specifically, the real-valued solution to $M(b) = C$ is never within one part in a million of an integer, $(\text{bandwidth\_gb\_per\_s} \cdot 10^9) / \text{kv\_bytes\_read}$ is never within one part in a million of $C$, and any existing crossover is at most `100,000`. Any correct double-precision implementation, including a batch-by-batch search, produces the same result.

## Why this matters

For a single stream, memory bandwidth usually sets decode speed rather than compute. Dividing bandwidth by model size predicts tokens per second on a laptop or CPU server within a small factor, and explains why a 4-bit model decodes roughly twice as fast as an 8-bit model on the same hardware.

Batching drives serving economics. Because weights are read once per step for the entire batch, each additional sequence adds little memory traffic until either KV caches dominate or the compute units saturate. Knowing which side of the crossover a deployment sits on tells you whether the next speedup comes from smaller weights, a smaller KV cache, a larger batch, or a faster processor.
