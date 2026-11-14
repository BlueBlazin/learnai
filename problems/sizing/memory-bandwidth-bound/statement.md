Compute the ceiling that memory bandwidth puts on decode speed, and the batch size at
which that ceiling stops being the limit. Generating one token touches every weight,
so the weights must cross the memory bus once per step however fast the arithmetic
is; what makes the problem more than one division is what batching does and does not
share.

## Specification

```python
def decode_tokens_per_s(weight_bytes: int, kv_bytes_read: int,
                        bandwidth_gb_per_s: float, batch: int) -> float: ...
def crossover_batch(weight_bytes: int, kv_bytes_read: int,
                    bandwidth_gb_per_s: float,
                    compute_tokens_per_s: float) -> int | None: ...
```

**The model.** A decode step advances `batch` sequences by one token each.

- The step reads every weight exactly once, **whatever the batch**: a weight matrix is
  streamed through once and applied to all sequences' activations. That is
  `weight_bytes` bytes.
- The step reads each sequence's KV cache once: `kv_bytes_read` bytes **per
  sequence**. All sequences have the same cache size.
- Nothing else is counted. The bytes read per step are therefore

$$
\text{bytes}(b) = \text{weight\_bytes} + b \cdot \text{kv\_bytes\_read}.
$$

- Memory delivers at most `bandwidth_gb_per_s` gigabytes per second, where **1 GB is
  $10^9$ bytes**, the unit hardware vendors quote. A step cannot take less time than
  its bytes take to arrive.

**`decode_tokens_per_s(weight_bytes, kv_bytes_read, bandwidth_gb_per_s, batch)`**
returns the resulting upper bound on throughput, in tokens per second **summed over
the batch** (one step emits `batch` tokens):

$$
M(b) = \frac{b \cdot \text{bandwidth\_gb\_per\_s} \cdot 10^9}{\text{weight\_bytes} + b \cdot \text{kv\_bytes\_read}} .
$$

It returns a float. The rate of a single sequence is $M(b) / b$.

**`crossover_batch(weight_bytes, kv_bytes_read, bandwidth_gb_per_s,
compute_tokens_per_s)`**. `compute_tokens_per_s` is the throughput $C$, summed over
the batch, that the processor's arithmetic could deliver if memory were infinitely
fast; in this model it does not depend on the batch. Real throughput is at most
$\min(M(b), C)$: decoding is *memory-bound* while $M(b) < C$ and *compute-bound* once
$M(b) \ge C$. The function returns the smallest integer batch $b \ge 1$ with
$M(b) \ge C$, as a Python int, or `None` if no batch satisfies it.

`None` is a real outcome: with `kv_bytes_read > 0`, $M(b)$ increases with $b$ but
stays below `bandwidth / kv_bytes_read`.

## Example

The 8-bit MLX build of Granite 4.2 3B
(`lmstudio-community/granite-4.2-3b-MLX-8bit`) is a single 3,888,771,048-byte file.
Take a machine with 400 GB/s of unified memory bandwidth, the figure Apple specifies
for the M1 Max and M2 Max. At a context of 4096 tokens one sequence's KV cache in
BF16 is $81920 \cdot 4096 = 335{,}544{,}320$ bytes.

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

A single stream cannot exceed $400 \cdot 10^9 / 3{,}888{,}771{,}048 = 102.9$ tokens
per second on this machine even with an empty cache, and 94.7 once it also reads a
4096-token cache. A batch of 8 shares one pass over the weights:
$8 \cdot 400 \cdot 10^9 / (3{,}888{,}771{,}048 + 8 \cdot 335{,}544{,}320) = 486.8$
tokens per second in total, 5.1 times the single stream rather than 8 times, because
the caches are not shared.

If the processor could compute 800 tokens per second, the memory bound reaches that
at batch 24 ($M(23) = 792.7$, $M(24) = 803.9$). At 1200 tokens per second it never
does: $M(b)$ approaches $400 \cdot 10^9 / 335{,}544{,}320 = 1192.1$ from below.

## Constraints

- `1 <= weight_bytes <= 10**13`, `0 <= kv_bytes_read <= 10**12`,
  `0 < bandwidth_gb_per_s <= 10**5`, `1 <= batch <= 10**6`,
  `0 < compute_tokens_per_s <= 10**9`.
- `decode_tokens_per_s` is compared with the reference to a relative tolerance of
  `1e-9`.
- `crossover_batch` is compared exactly. The tests avoid inputs on a knife edge: the
  real-valued batch at which $M(b) = C$ is never within one part in a million of an
  integer, `bandwidth / kv_bytes_read` is never within one part in a million of $C$,
  and when a crossover exists it is at most 100,000. Any correct double-precision
  implementation therefore gets the same answer, including one that searches batch
  by batch.

## Why this matters

For a single stream, memory bandwidth, not arithmetic, usually sets decode speed:
"model bytes divided into bandwidth" predicts tokens per second on a laptop or a CPU
server to within a small factor, and it explains at a glance why a 4-bit build
decodes about twice as fast as an 8-bit one on the same machine. The batch term is
the economics of serving: the weights are read once per step for everyone, so each
added sequence is nearly free until either the caches dominate the traffic or the
arithmetic units are finally busy. Knowing which side of the crossover a deployment
sits on tells you whether the next improvement comes from smaller weights, a smaller
cache, a larger batch, or a faster processor.
