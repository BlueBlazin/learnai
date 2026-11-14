## The idea

To produce a token, a decoder multiplies activations by every weight matrix of the
model. The weights do not fit in any processor cache (gigabytes against megabytes),
so each step streams all of them from main memory, and a step cannot finish before
its bytes have arrived:

$$
\text{step time} \ge \frac{\text{bytes read}}{\text{bandwidth}} .
$$

For one sequence the bytes are the weights plus that sequence's KV cache, which
attention reads in full. So a single stream runs at no more than
`bandwidth / (W + K)` tokens per second. Nothing about the processor's arithmetic
speed appears in the bound: the decode step of the example, 8.5 GFLOP of arithmetic
(see `flops-per-token`), takes at least 10.6 milliseconds on that machine however
fast the arithmetic is.

**Batching.** With `b` sequences, each weight matrix is still read once and applied
to a `b`-row activation matrix; the caches are private. Bytes per step are `W + bK`,
and the step yields `b` tokens:

$$
M(b) = \frac{bB}{W + bK}, \qquad B = \text{bandwidth in bytes per second}.
$$

Two regimes follow. While $bK \ll W$ the weights dominate and throughput grows almost
linearly with the batch: sequences ride along for free. As $b \to \infty$ the caches
dominate and $M(b) \to B / K$: the ceiling for long contexts is set by the cache, not
by the model size. Per-sequence speed $M(b)/b = B / (W + bK)$ only ever goes down.

**The crossover.** The processor can compute at most $C$ tokens per second.
Throughput is $\min(M(b), C)$, and the regime changes at the smallest $b$ with
$M(b) \ge C$:

$$
\frac{bB}{W + bK} \ge C \iff b\,(B - CK) \ge CW \iff b \ge \frac{CW}{B - CK}
\quad\text{if } B > CK .
$$

If $B \le CK$, the left side of the middle inequality is not positive while the right
side is, so no batch qualifies: the bound saturates at $B/K \le C$ and decoding stays
memory-bound at every batch. Otherwise the answer is the ceiling of the threshold,
which is at least 1 because the threshold is positive.

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

- **Weights.** `lmstudio-community/granite-4.2-3b-MLX-8bit` on the Hugging Face Hub
  lists `model.safetensors` at 3,888,771,048 bytes (read on 2026-11-14). The BF16
  original, `ibm-granite/granite-4.2-3b`, is 7,319,517,120 bytes in two shards; the
  same machine bounds it at 52.3 tokens per second with the 4096-token cache.
- **Cache.** Granite 4.2 3B has 40 layers and 8 KV heads of width 64
  (`config.json`), so one token is $2 \cdot 40 \cdot 8 \cdot 64 \cdot 2 = 81{,}920$
  bytes in BF16 (`kv-cache-bytes`).
- **Bandwidth.** Apple's announcement of the M2 Pro and M2 Max (Apple Newsroom,
  January 2023) gives the M2 Max "400GB/s of unified memory bandwidth", twice the M2
  Pro's 200 GB/s; the M1 Max was specified the same way. Vendor figures are decimal
  and are peak rates.

## Common mistakes

- **Charging the weights once per sequence.** `batch * (W + K)` bytes per step makes
  throughput independent of the batch, as if eight requests were eight separate
  model runs. It is exactly right at batch 1, which is why it survives a
  single-stream benchmark, and it hides the whole reason servers batch.
- **Ignoring the KV cache.** `B / W` is the familiar rule and it is fine for short
  contexts. At 4096 tokens the cache is 8 percent of this model's bytes at batch 1
  and 41 percent at batch 8, and without it the bound grows forever: the function
  can then never return `None`.
- **GiB for GB.** Bandwidth is quoted in decimal units; file sizes are often shown in
  binary ones. Converting 400 GB/s with $2^{30}$ gives 429.5 billion bytes per second,
  7.4 percent too many. The reverse mixup (a "3.62 GiB" file treated as 3.62 GB)
  costs the same 7.4 percent.
- **Returning the per-sequence rate.** `B / (W + bK)` is steps per second, which is
  what one user sees. The total is `b` times that. Both are useful; they answer
  different questions, and a capacity plan needs the total.
- **Rounding the crossover down.** `int(x)` gives the last batch that is still
  memory-bound. The smallest batch that *reaches* the compute rate is the ceiling.
- **Dividing by a non-positive number.** When $B \le CK$ the formula produces a
  negative or infinite threshold. Clamping it to 1 reports "compute-bound from the
  first sequence" for the one case that is memory-bound at every batch.

## Complexity

$O(1)$. A linear search over the batch is also acceptable under the constraints, but
it needs the saturation test anyway to know when to stop.

## In the wild

- **It is a ceiling.** Real decoders land below it: quoted bandwidth is a peak, some
  bytes are read more than once, and time is also spent on arithmetic, sampling, and
  scheduling. They do not land above it, so the bound is the quickest way to spot a
  benchmark result that cannot be right.
- **What `W` should be.** Bytes actually read per step. The token embedding table is
  indexed, not streamed, so it can be left out (7 percent of Granite 4.2 3B's
  parameters; the untied output head, by contrast, is read in full). Sparse
  architectures change the count more: a mixture-of-experts model reads only the
  experts a token is routed to.
- **Quantization is a speed feature here.** Halving the bytes per weight nearly
  doubles the single-stream bound, independent of any change in arithmetic cost.
- **GPUs follow the same law with different constants.** Their bandwidth is several
  times higher and their compute rate far higher still, so they sit deep in the
  memory-bound regime at batch 1 and need large batches to reach the crossover. That
  gap is the capacity that continuous batching sells.
