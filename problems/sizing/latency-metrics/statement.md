Compute five latency and throughput metrics for a single streamed LLM request from its send time and token arrival timestamps.

## Specification

```python
def request_metrics(t_start: float, token_times: list[float]) -> dict: ...
```

- `t_start` is the time when the client sent the request, in seconds.
- `token_times` holds the arrival time of each output token, in seconds on the same clock, in arrival order. Write $n$ for its length and $t_1 \le t_2 \le \dots \le t_n$ for its values. Timestamps never decrease, and $t_1 \ge t_\text{start}$. Consecutive timestamps may be equal when several tokens arrive in one chunk.

The clock origin is arbitrary. `t_start` may be `0.0`, a monotonic clock reading, or a Unix timestamp, and only differences between timestamps matter.

Return a dictionary with exactly these five keys:

| key | value | definition |
|---|---|---|
| `"ttft"` | `float` | time to first token, $t_1 - t_\text{start}$ |
| `"itl"` | list of $n - 1$ `float`s | inter-token latencies in arrival order, $t_{i+1} - t_i$ for $i = 1, \dots, n - 1$ |
| `"tpot"` | `float` or `None` | time per output token, $\dfrac{t_n - t_1}{n - 1}$ |
| `"e2e"` | `float` | end-to-end latency, $t_n - t_\text{start}$ |
| `"tokens_per_s"` | `float` or `None` | output tokens per second, $\dfrac{n}{t_n - t_\text{start}}$ |

Handle these edge cases:

- **One token** ($n = 1$). Set `"itl"` to `[]` and `"tpot"` to `None`. A single token has no inter-token gap to measure, whereas returning `0.0` would falsely report a zero-duration decode step.
- **Zero end-to-end latency** ($t_n = t_\text{start}$, which a coarse clock can produce). Set `"tokens_per_s"` to `None`. Compute all other keys from their definitions, so `"ttft"` and `"e2e"` are `0.0`.
- **No tokens** ($n = 0$). Raise `ValueError`.

Do not modify `token_times`. Return Python `float` values for all numeric scalars and elements of `"itl"`.

> [!NOTE]
> **`"tpot"` excludes the first token** in both the elapsed time and the token count. This matches NVIDIA AIPerf's inter-token definition, `(request_latency - time_to_first_token) / (output_tokens - 1)`, and equals the mean of `"itl"`. GuideLLM uses the name TPOT for a different formula that includes the first token, $\dfrac{t_n - t_\text{start}}{n}$, and calls the formula above ITL. Implement the first definition here.

## Example

A request is sent at `t_start = 12.0` and five tokens arrive:

```python
request_metrics(12.0, [12.5, 12.5625, 12.625, 12.75, 13.0])
```

This call returns:

```python
{
    "ttft": 0.5,                              # 12.5 - 12.0
    "itl": [0.0625, 0.0625, 0.125, 0.25],     # four gaps between five tokens
    "tpot": 0.125,                            # (13.0 - 12.5) / 4
    "e2e": 1.0,                               # 13.0 - 12.0
    "tokens_per_s": 5.0,                      # 5 / 1.0
}
```

![The example request. TTFT is the wait for the first token; TPOT is the span from $t_1$ to $t_n$ divided by $n - 1$.](assets/latency-timeline.svg)

Half of this 1.0 s request was spent waiting for the first token. Different tools summarize the same stream with different formulas:

| quantity | formula | value |
|---|---|---|
| `"tpot"` (this problem, AIPerf's inter-token latency) | $(t_n - t_1) / (n - 1)$ | 0.125 s |
| TPOT including the first token (GuideLLM) | $(t_n - t_\text{start}) / n$ | 0.2 s |
| `"tokens_per_s"` (this problem, end to end) | $n / (t_n - t_\text{start})$ | 5 tokens/s |
| decode speed, $1 / \text{tpot}$ | $(n - 1) / (t_n - t_1)$ | 8 tokens/s |

With one token, `request_metrics(3.0, [3.75])` returns `{"ttft": 0.75, "itl": [], "tpot": None, "e2e": 0.75, "tokens_per_s": 1.3333333333333333}`.

## Constraints

- `0 <= len(token_times) <= 100,000`, with finite floats in $[0, 2 \cdot 10^9]$.
- Floats are compared with a relative tolerance of `1e-9` and an absolute tolerance of `1e-9`. `None`, the dictionary keys, and the length of `"itl"` are compared exactly.
- Use plain Python. There is no speed requirement beyond the time limit.

## Why this matters

A streamed LLM request has two distinct phases. Time to first token (TTFT) captures queueing and prompt prefill, so it grows with server load and prompt length and determines perceived responsiveness. Inter-token latency (ITL) captures the decode loop, which depends on batch size and model speed and determines how smoothly text streams.

Folding the first token into the per-token average mixes those two phases together. A long prompt then looks like a slow decoder, and numbers from two benchmark tools cannot be compared even when they share a label. Because LLM serving SLOs target these exact quantities (such as p99 TTFT under 500 ms and TPOT under 50 ms), every benchmark report must be explicit about which time interval and token count it divides.
