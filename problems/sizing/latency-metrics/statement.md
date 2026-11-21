Compute the latency metrics of one streamed request from its timestamps: time to first
token, the gaps between tokens, time per output token, end-to-end latency, and tokens
per second. Each is one subtraction or one division; the substance is which interval
and which token count each metric uses, because tools that share these names do not
share the definitions.

## Specification

```python
def request_metrics(t_start: float, token_times: list[float]) -> dict: ...
```

- `t_start` is the time at which the client sent the request, in seconds.
- `token_times` holds the arrival time of each output token, in seconds on the same
  clock, in arrival order. Write $n$ for its length and $t_1 \le t_2 \le \dots \le t_n$
  for its values. Times never decrease, and $t_1 \ge t_\text{start}$. Consecutive
  times may be equal (several tokens delivered in one chunk).

The clock is arbitrary: `t_start` may be 0, a monotonic clock reading, or a Unix
timestamp. Only differences matter.

The function returns a dict with exactly these five keys:

| key | value | definition |
|---|---|---|
| `"ttft"` | float | time to first token, $t_1 - t_\text{start}$ |
| `"itl"` | list of $n - 1$ floats | inter-token latencies in arrival order: $t_{i+1} - t_i$ for $i = 1, \dots, n-1$ |
| `"tpot"` | float or `None` | time per output token, $\dfrac{t_n - t_1}{n - 1}$ |
| `"e2e"` | float | end-to-end latency, $t_n - t_\text{start}$ |
| `"tokens_per_s"` | float or `None` | output tokens per second, $\dfrac{n}{t_n - t_\text{start}}$ |

Edge cases:

- **One token** ($n = 1$): `"itl"` is the empty list and `"tpot"` is `None`. A single
  token has no gap to measure; `0.0` would claim that a decode step took no time.
- **Zero end-to-end latency** ($t_n = t_\text{start}$, which a coarse clock can
  produce): `"tokens_per_s"` is `None`. The other values follow their definitions
  (`"ttft"` and `"e2e"` are `0.0`).
- **No tokens**: an empty `token_times` raises `ValueError`.

The function does not modify `token_times`. Scalars are Python floats.

> [!NOTE]
> **TPOT excludes the first token**, in the time and in the count. This is the
> inter-token definition that NVIDIA's AIPerf uses:
> `(request_latency - time_to_first_token) / (output_tokens - 1)`. It equals the mean
> of `"itl"`. GuideLLM reports a metric with the same name that *includes* the first
> token, $\dfrac{t_n - t_\text{start}}{n}$, and calls the definition above ITL. This
> problem's `"tpot"` is the first one; the example below shows both numbers.

## Example

A request is sent at `t_start = 12.0` and five tokens arrive:

```python
request_metrics(12.0, [12.5, 12.5625, 12.625, 12.75, 13.0])
```

returns

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

Half of this request's second was spent waiting for the first token. The three ways of
averaging tell three stories about the same stream:

| quantity | formula | value |
|---|---|---|
| `"tpot"` (this problem, AIPerf's inter-token latency) | $(t_n - t_1) / (n - 1)$ | 0.125 s |
| TPOT including the first token (GuideLLM) | $(t_n - t_\text{start}) / n$ | 0.2 s |
| `"tokens_per_s"` (this problem, end to end) | $n / (t_n - t_\text{start})$ | 5 tokens/s |
| decode speed, $1 / \text{tpot}$ | $(n - 1) / (t_n - t_1)$ | 8 tokens/s |

With one token, `request_metrics(3.0, [3.75])` returns
`{"ttft": 0.75, "itl": [], "tpot": None, "e2e": 0.75, "tokens_per_s": 1.3333333333333333}`.

## Constraints

- `0 <= len(token_times) <= 100,000`; times are finite floats in $[0, 2 \cdot 10^9]$.
- Floats are compared with a relative tolerance of `1e-9` and an absolute tolerance
  of `1e-9`; `None`, the keys, and the length of `"itl"` are compared exactly.
- Use plain Python; no speed requirement beyond the time limit.

## Why this matters

A streaming request has two phases with different costs and different owners. The time
to first token is queueing plus prefill: it grows with load and with prompt length, and
it is what a user perceives as responsiveness. The time between tokens is the decode
loop: it depends on batch size and model speed, and it is what makes a stream feel
smooth. A metric that folds the first token into the per-token average mixes the two,
so a long prompt looks like a slow decoder, and a number from one benchmark tool cannot
be compared with the same-named number from another. Service-level objectives for LLM
serving are written against exactly these quantities ("p99 TTFT under 500 ms, TPOT
under 50 ms"), so a benchmark report has to say which interval and which count it
divided.
