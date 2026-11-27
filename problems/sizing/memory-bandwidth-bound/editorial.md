## The idea

Let $W$ be `weight_bytes`, $K$ be `kv_bytes_read`, and $B = \text{bandwidth\_gb\_per\_s} \cdot 10^9$ be the memory bandwidth in bytes per second. To generate a token, a decoder multiplies activations by every weight matrix in the model. Because model weights (gigabytes) do not fit in processor caches (megabytes), each decode step streams all weights from main memory and cannot finish before those bytes arrive.

$$
\text{step time} \ge \frac{\text{bytes read}}{B} .
$$

For a single sequence, one step reads the weights plus that sequence's KV cache, bounding throughput at $B / (W + K)$ tokens per second. Compute speed does not appear in this bound. In the Granite 4.2 3B example, a decode step performs 8.5 GFLOP of arithmetic (see `flops-per-token`), yet still takes at least 10.6 milliseconds on a 400 GB/s machine regardless of how fast the arithmetic units are.

**Batching.** With a batch of $b$ sequences, each weight matrix is still read once and applied to a $b$-row activation matrix, while each sequence reads its own KV cache. One step transfers $W + bK$ bytes and produces $b$ tokens, so total throughput is bounded by

$$
M(b) = \frac{bB}{W + bK} .
$$

Two regimes emerge from this formula. While $bK \ll W$, weights dominate and total throughput scales nearly linearly with $b$ because extra sequences share the weight read. As $b \to \infty$, KV caches dominate and $M(b)$ approaches $B / K$, so long-context throughput is capped by cache reads rather than model size. Meanwhile, per-sequence speed $M(b)/b = B / (W + bK)$ decreases monotonically with $b$.

**The crossover.** Let $C$ be `compute_tokens_per_s`. Real throughput is at most $\min(M(b), C)$, and decoding transitions from memory-bound to compute-bound at the smallest integer $b \ge 1$ where $M(b) \ge C$. Rearranging the inequality gives

$$
\frac{bB}{W + bK} \ge C \iff b\,(B - CK) \ge CW \iff b \ge \frac{CW}{B - CK}
\quad\text{if } B > CK .
$$

If $B \le CK$, the left-hand side of $b(B - CK) \ge CW$ is non-positive while the right-hand side is strictly positive, so no batch qualifies. In that case, $M(b)$ saturates at $B / K \le C$ and decoding remains memory-bound at every batch size, so the function returns `None`. When $B > CK$, the answer is $\lceil CW / (B - CK) \rceil$, which is always at least 1 because $CW / (B - CK) > 0$.

## Reference solution (Python)

```python
import math

GB = 10**9


def decode_tokens_per_s(weight_bytes, kv_bytes_read, bandwidth_gb_per_s, batch):
    bytes_per_step = weight_bytes + batch * kv_bytes_read
    steps_per_s = bandwidth_gb_per_s * GB / bytes_per_step
    return batch * steps_per_s


def crossover_batch(weight_bytes, kv_bytes_read, bandwidth_gb_per_s, compute_tokens_per_s):
    bandwidth = bandwidth_gb_per_s * GB
    spare = bandwidth - compute_tokens_per_s * kv_bytes_read
    if spare <= 0:
        return None
    return math.ceil(compute_tokens_per_s * weight_bytes / spare)
```

## Where the numbers come from

- **Weights.** `lmstudio-community/granite-4.2-3b-MLX-8bit` on the Hugging Face Hub lists `model.safetensors` at 3,888,771,048 bytes (read on 2026-11-14). The BF16 original, `ibm-granite/granite-4.2-3b`, is 7,319,517,120 bytes across two shards, which bounds the same machine at 52.3 tokens per second with a 4096-token cache.
- **Cache.** Granite 4.2 3B has 40 layers and 8 KV heads of width 64 (`config.json`), so one token uses $2 \cdot 40 \cdot 8 \cdot 64 \cdot 2 = 81{,}920$ bytes in BF16 (see `kv-cache-bytes`).
- **Bandwidth.** Apple's announcement of the M2 Pro and M2 Max (Apple Newsroom, January 2023) specifies 400 GB/s of unified memory bandwidth for the M2 Max (twice the M2 Pro's 200 GB/s), matching the M1 Max. Vendor bandwidth specs are decimal peak rates.

## Common mistakes

- **Charging weights once per sequence.** Using `batch * (W + K)` bytes per step makes throughput independent of batch size, as if eight requests ran the model separately. It agrees at `batch = 1` and therefore passes single-stream checks, while missing the entire benefit of batching.
- **Ignoring the KV cache.** Using `B / W` works for short contexts, but at 4096 tokens the cache is 8 percent of traffic at batch 1 and 41 percent at batch 8. Without the cache term, $M(b)$ grows without bound and `crossover_batch` can never return `None`.
- **Using GiB instead of GB.** Memory bandwidth is specified in decimal gigabytes ($10^9$ bytes), whereas file sizes are often reported in binary gibibytes ($2^{30}$ bytes). Multiplying 400 GB/s by $2^{30}$ yields 429.5 billion bytes per second, an error of 7.4 percent. Treating a 3.62 GiB file as 3.62 GB introduces the same 7.4 percent error in reverse.
- **Returning per-sequence throughput.** Computing `B / (W + bK)` gives steps per second, which is the rate a single user observes. Total throughput is `batch` times that rate.
- **Rounding the crossover down.** Taking `int(x)` returns the largest batch that is still memory-bound. The smallest batch that reaches $C$ is `math.ceil(x)`.
- **Dividing by a non-positive denominator.** When $B \le CK$, `CW / (B - CK)` is negative or undefined. Clamping negative values to 1 wrongly reports that decoding is compute-bound at batch 1 when it is actually memory-bound at every batch size.

## Complexity

$O(1)$ time and space. A linear search over batch sizes also passes under the constraints, provided it still checks $B \le CK$ first so it knows when to return `None`.

## In the wild

- **A hard upper bound.** Real decoders fall below this ceiling because quoted bandwidth is a peak rate, some data is read more than once, and steps also spend time on arithmetic, sampling, and scheduling. Because real runs cannot exceed it, the bound is a fast sanity check for suspicious benchmark numbers.
- **Choosing `W`.** Count only bytes actually streamed per step. The token embedding table is indexed rather than streamed and can be omitted (7 percent of Granite 4.2 3B's parameters), whereas an untied output head is read in full. Mixture-of-experts models stream only the experts selected for each token.
- **Quantization improves decode speed.** Halving the bytes per weight nearly doubles single-stream decode speed even if arithmetic throughput does not change.
- **GPUs obey the same law with different constants.** GPUs have higher memory bandwidth and much higher compute throughput than laptops, so they sit deep in the memory-bound regime at batch 1 and require large batches to reach the crossover. Continuous batching exists to capture that headroom.
