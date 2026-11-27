## The idea

Place the request on a timeline. The send time $t_\text{start}$ and the $n$ token arrival times divide the timeline into $n$ intervals. The first interval runs from $t_\text{start}$ to $t_1$, and the remaining $n - 1$ intervals run between consecutive tokens.

- **TTFT** is the first interval, $t_1 - t_\text{start}$. It covers network transit, queueing, and prompt prefill, and it produces the first token.
- **ITL** is the list of the other $n - 1$ intervals, $t_{i+1} - t_i$. Each interval is one decode step, or zero when multiple tokens arrive in the same chunk.
- **TPOT** is the mean inter-token latency. Because the sum of consecutive gaps telescopes, $\sum_{i=1}^{n-1} (t_{i+1} - t_i) = t_n - t_1$, TPOT simplifies to $(t_n - t_1) / (n - 1)$. Both the first interval and the first token are excluded. Dropping only one of them produces a ratio that is not the mean of any set of intervals.
- **End-to-end latency** covers all $n$ intervals, $t_n - t_\text{start}$. **Tokens per second** divides all $n$ tokens by that full duration to measure the rate the user experienced.

The identity $\text{e2e} = \text{ttft} + (n - 1)\cdot\text{tpot}$ links these quantities and serves as a quick sanity check on any latency report.

The two `None` cases occur when a ratio has a zero denominator. One token has no gap to average, and a request with zero elapsed time has no defined rate. Returning `0.0` or infinity instead would inject fabricated values into downstream means and percentiles.

## Reference solution (Python)

```python
def request_metrics(t_start, token_times):
    n = len(token_times)
    if n == 0:
        raise ValueError("a request without output tokens has no latency metrics")
    first, last = token_times[0], token_times[-1]
    e2e = last - t_start
    return {
        "ttft": first - t_start,
        "itl": [token_times[i] - token_times[i - 1] for i in range(1, n)],
        "tpot": (last - first) / (n - 1) if n > 1 else None,
        "e2e": e2e,
        "tokens_per_s": n / e2e if e2e > 0 else None,
    }
```

## Common mistakes

- **Including the first token in TPOT (`e2e / n`).** This is GuideLLM's TPOT, not the definition required here. With a long prompt, prefill dominates the average. In the worked example, it reports 0.2 s per token for a decoder taking 0.125 s per token.
- **Dividing by `n` instead of `n - 1`.** Five tokens have four gaps. The error is 20 percent at $n = 5$ and nearly invisible at $n = 500$, so it often goes unnoticed in benchmark scripts.
- **Including `t_start` at the start of `"itl"`.** Prepending `t_start` makes TTFT the first inter-token latency. Because TTFT is often 10 to 100 times larger than decode gaps, it corrupts the p99 of the ITL distribution.
- **Computing `tokens_per_s` over generation time only (`n / (t_n - t_1)`).** This hides the wait for the first token and counts `n` tokens over `n - 1` gaps. It measures decode speed rather than end-to-end user throughput.
- **Returning `0.0` for single-token TPOT or infinity for zero-duration throughput.** Both invent measurements that were never observed.
- **Using a raw timestamp as a duration.** Writing `ttft = token_times[0]` passes toy tests where `t_start == 0` and fails on real clocks.
- **Sorting `"itl"`.** The list is a time series, so keeping arrival order preserves where stalls occurred.
- **Returning metrics for an empty `token_times`.** A request with zero tokens has no first token to time and should raise `ValueError` so callers count it as an error or empty response.

## Complexity

$O(n)$ time and $O(n)$ memory to construct `"itl"`. Computing the scalar metrics takes $O(1)$ time.

## In the wild

- **AIPerf** (NVIDIA, successor to GenAI-Perf) defines TTFT as the first response timestamp minus the request start time, and inter-token latency as `(request_latency - time_to_first_token) / (output_sequence_length - 1)` to exclude TTFT overhead. Its per-user output throughput is the reciprocal of that value, which is 8 tokens/s in the worked example ([AIPerf metrics reference](https://docs.nvidia.com/aiperf/latest/reference/ai-perf-metrics-reference)).
- **GuideLLM** (vLLM project) reports both metrics: an inter-token latency that excludes the first token and a time per output token that includes it ([GuideLLM metrics guide](https://github.com/vllm-project/guidellm)). Two reports labeled "TPOT" can therefore differ by the entire prefill time.
- **Server-side metrics** measure the same phases from inside the engine. vLLM exports TTFT and inter-token histograms on `/metrics` (including `vllm:time_to_first_token_seconds`). Because server-side timers exclude network transit and client-side queueing, they are lower bounds on client-observed latency.
- **Chunks are not tokens.** A streamed SSE chunk can carry several tokens, and detokenization can buffer a token until the next one completes a multi-byte character. A client that timestamps chunks measures inter-chunk latency, which AIPerf tracks separately. Per-token timestamps require either one token per chunk or server-reported usage counts.
- **System throughput across requests** is total output tokens divided by the benchmark's total wall-clock duration. It is not the mean of per-request `tokens_per_s`, because rates over different durations do not average directly.
