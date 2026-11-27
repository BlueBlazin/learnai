## The idea

Every latency metric is a difference between two clock readings, so correctness depends on *when* you read the clock. Consider the awaits in a single request:

```
probe called        response begins      chunk arrives       sink returns
     |---- open_stream() ----|---- next chunk ----|---- sink(chunk) ----|---- next chunk ...
  t_start                                      arrived
```

- Read `t_start` before `open_stream()`. From the user's perspective, waiting begins when the request is sent. The TCP handshake, server admission queue, and prefill are all part of waiting for the first token. Reading the clock after `open_stream()` drops everything before the response headers arrive.
- Read `arrived` between receiving the chunk and calling `sink`. Work inside `sink` belongs to the client, not the server. Reading the clock after `sink` adds client processing time to TTFT and to every inter-token gap.

**Chunks are not tokens.** A stream delivers messages, while latency metrics are defined on tokens. A chunk with $k$ tokens represents $k$ tokens arriving together, so it adds $k$ identical timestamps (producing $k - 1$ zero-length gaps). An empty chunk adds nothing and cannot set TTFT or E2E. Under this rule, `(t_n - t_1) / (n - 1)` gives the true mean time per token regardless of server batching, and TTFT ignores role-only opening chunks sent by OpenAI-compatible servers.

**Failures are not fast requests.** A failed or empty stream records `None` for its latency metrics, and `summarize` excludes non-`"ok"` records. Recording zero or the time of failure would inject fast samples into the distribution, making percentiles improve as the server fails more often.

**Pooled gaps.** The ITL percentile aggregates every inter-token gap across all `"ok"` requests. Taking a percentile of per-request means gives TPOT instead. A stream that stalls once for two seconds while remaining fast elsewhere can still have a normal mean, and p99 ITL exists specifically to catch that stall.

**The closed loop** keeps `concurrency` requests in flight by launching a new request the instant another completes. A pool of `min(concurrency, n)` worker coroutines sharing a counter achieves this cleanly. A task per request gated by an `asyncio.Semaphore` also works, since waiters wake in FIFO order.

## Reference solution (Python)

```python
async def probe(open_stream, sink):
    t_start = now()
    token_times, status = [], "ok"
    try:
        stream = await open_stream()
        async for chunk in stream:
            arrived = now()
            token_times.extend([arrived] * len(chunk))
            await sink(chunk)
    except OSError:
        status = "error"
    else:
        if not token_times:
            status = "empty"

    result = {
        "status": status,
        "t_start": t_start,
        "token_times": token_times,
        "ttft": None,
        "itl": [],
        "tpot": None,
        "e2e": None,
    }
    if status == "ok":
        n, first, last = len(token_times), token_times[0], token_times[-1]
        result["ttft"] = first - t_start
        result["itl"] = [b - a for a, b in pairwise(token_times)]
        result["tpot"] = (last - first) / (n - 1) if n > 1 else None
        result["e2e"] = last - t_start
    return result
```

Catching `OSError` handles transport failures such as refused connections, resets, and timeouts (`ConnectionError` and `TimeoutError` are both subclasses of `OSError`). Programming errors like `TypeError` still propagate, and `asyncio.CancelledError` passes through so a canceled benchmark stops cleanly.

## Common mistakes

- **Starting the clock after the first await.** Calling `t_start = now()` after `await open_stream()` drops connection and queueing time from TTFT and E2E (yielding 0.5 s instead of 0.75 s in the example).
- **Stamping after processing.** Calling `await sink(chunk)` before `now()` adds client delay to every timestamp. In production clients, the same bug happens when parsing JSON, detokenizing, or printing before reading the clock.
- **Counting a failed stream as a success.** Catching `OSError` and falling through to the normal computation marks a broken stream as `"ok"` using partial tokens, or as zero latency if no tokens arrived.
- **Measuring TTFT at the first chunk.** Chat streams commonly start with a role-only chunk that has no content. That chunk arrives as soon as the server accepts the request, well before prefill finishes.
- **Recording one timestamp per chunk.** Treating chunks as tokens inflates TPOT by the server's batching factor.
- **Measuring E2E at stream end.** The closing usage chunk and connection teardown happen after the user has already received the last token.
- **Using nearest-rank instead of linear interpolation.** Both percentile definitions are common, and they disagree on small samples. In the example, p50 TTFT is 0.375 with linear interpolation and 0.25 with nearest-rank.
- **Taking a percentile of per-request mean ITLs.** Per-request means measure TPOT, not pooled ITL.
- **Including failed records in the summary.** Substituting `0` for `None` pulls every percentile downward.
- **Running requests in waves.** Calling `asyncio.gather` on fixed batches of size `concurrency` waits for the slowest request in each wave before starting the next. In the example, request 2 would start at 1.0 s instead of 0.5 s, underloading the server.
- **Returning results in completion order.** Appending records as probes finish scrambles the output. Store each record at its request index.
- **Using the real clock.** `time.monotonic()` matches the event loop clock in production but breaks under simulated time, where tests can execute in microseconds.

## Complexity

- `probe`: $O(K)$ time and memory for a stream with $K$ total tokens.
- `summarize`: $O(G \log G)$ time to sort $G$ pooled gaps.
- `run_closed_loop`: $O(n)$ orchestration overhead on top of the $n$ probes.

## In the wild

- **vLLM's benchmark client** (`vllm/benchmarks/lib/endpoint_request_func.py`) reads `time.perf_counter()` for each message before parsing it, sets TTFT on the first chunk carrying output (skipping role-only opening chunks), and records E2E on the last output chunk rather than the usage chunk. Unlike this problem, it records one ITL per chunk and reads the token count from the server's usage report, so its TPOT is `(latency - ttft) / (output_tokens - 1)` while its ITL list has one entry per message. It computes percentiles with `numpy.percentile` using linear interpolation.
- **Closed vs. open loop.** A closed loop with `concurrency` workers models N concurrent users. Its offered rate is throttled by the server's own speed, so when the server slows down, the client sends fewer requests and misses latencies it would have observed during the slowdown (coordinated omission). Use a closed loop to measure throughput at fixed concurrency, and an open loop (which sends on an independent schedule) to measure latency at a target arrival rate.
- **Slow sinks in practice.** A Python benchmark client that detokenizes and logs inside the receive loop can spend more time per chunk than a fast server takes to emit it. Incoming chunks then back up in the socket buffer, arrival times bunch together, and measured ITL reflects the client rather than the server. Stamping before processing and keeping the receive loop lightweight prevents this.
