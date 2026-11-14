## The idea

Place the request on a timeline. The send time and the $n$ token arrivals cut it into
$n$ intervals: one from the send to the first token, and $n - 1$ between tokens.

- The first interval is **TTFT**. It contains the network, the queue, and the prefill
  of the whole prompt, and it produces one token.
- The other $n - 1$ intervals are the **inter-token latencies**. Each is one decode
  step (or zero, when a chunk carries several tokens).
- **TPOT** is the mean inter-token latency. The sum of the gaps telescopes,
  $\sum_{i=1}^{n-1} (t_{i+1} - t_i) = t_n - t_1$, so TPOT is $(t_n - t_1)/(n - 1)$:
  the first interval is excluded from the time, and the token it produced is excluded
  from the count. Both exclusions are needed. Dropping only one gives a number that
  is not the mean of anything.
- **End-to-end latency** is all $n$ intervals, and **tokens per second** here is all
  $n$ tokens over all $n$ intervals: the rate the user experienced.

The identity $\text{e2e} = \text{ttft} + (n - 1)\cdot\text{tpot}$ ties them together
and is a useful check on any latency report.

The two `None` results are the cases where a ratio has no denominator: one token has no
gap, and a request with zero duration has no rate. Reporting `0.0` or infinity instead
would put a fabricated value into every average and percentile computed downstream.

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

- **Including the first token in TPOT**: `e2e / n`. This is a legitimate metric
  (GuideLLM's TPOT) but a different one. With a long prompt it is dominated by
  prefill: in the example it reports 0.2 s per token for a decoder that takes 0.125 s.
- **Dividing by `n` instead of `n - 1`.** Five tokens have four gaps. The error is
  20 percent for five tokens and invisible for five hundred, which is why it survives
  in benchmark scripts.
- **Starting the inter-token list at the request.** Prepending `t_start` makes the
  first "inter-token" latency the TTFT, which is often 10 to 100 times larger than
  the others and wrecks the p99 of the gap distribution.
- **Tokens per second over the generation time only.** `n / (t_n - t_1)` hides the
  wait for the first token (and counts `n` tokens over `n - 1` gaps). It is a decode
  speed, not what the user got. Whichever you report, say which.
- **`0.0` for the TPOT of a single token**, or infinity for the rate of a request with
  zero duration. Both are measurements that were never made.
- **Using a raw timestamp as a duration.** `ttft = token_times[0]` works in every toy
  example whose clock starts at 0.
- **Sorting the gaps.** The list is a time series: where the stall happened (after
  the third token, at a chunk boundary) is information.
- **Returning metrics for a request with no tokens.** There is no first token to time.
  Count such requests separately, as errors or empty responses.

## Complexity

$O(n)$ time and memory for the list of gaps; the scalars are $O(1)$.

## In the wild

- **AIPerf** (NVIDIA; the successor of GenAI-Perf) defines time to first token as the
  first response's timestamp minus the request start, and inter-token latency as
  `(request_latency - time_to_first_token) / (output_sequence_length - 1)`, "excluding
  the initial TTFT overhead". Its per-user output throughput is the reciprocal of that,
  the 8 tokens/s of the example
  ([AIPerf metrics reference](https://docs.nvidia.com/aiperf/latest/reference/ai-perf-metrics-reference)).
- **GuideLLM** (vLLM project) reports both: an inter-token latency that excludes the
  first token and a time per output token that includes it
  ([GuideLLM metrics guide](https://github.com/vllm-project/guidellm)). Two reports
  that both say "TPOT" can therefore differ by the whole prefill time.
- **Servers** measure the same intervals from the inside. vLLM exports histograms of
  time to first token and of the time between tokens on its `/metrics` endpoint
  (`vllm:time_to_first_token_seconds` and an inter-token histogram). Server-side
  values exclude the network and client-side queueing, so they are lower bounds on
  what a client measures.
- **Chunks are not tokens.** A streamed chunk can carry several tokens, and
  detokenization can hold a token back until the next one completes a character. A
  client that timestamps chunks measures inter-chunk latency; AIPerf keeps that as a
  separate metric. Timestamps per token need either one token per chunk or the
  server's own usage counts.
- **Throughput across requests** is a different quantity again: total output tokens
  divided by the benchmark's duration. It is not the mean of per-request
  `tokens_per_s`, for the same reason percentiles do not average.
