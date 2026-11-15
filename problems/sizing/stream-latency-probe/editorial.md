## The idea

A latency is the difference of two clock readings, so the whole problem is *when the
clock is read*. Mark the awaits of one request on a timeline:

```
probe called        response begins      chunk arrives       sink returns
     |---- open_stream() ----|---- next chunk ----|---- sink(chunk) ----|---- next chunk ...
  t_start                                      arrived
```

- `t_start` goes before `open_stream()`. From the user's side the request began when
  they pressed enter; the TCP handshake, the server's admission queue, and the prefill
  are all "waiting for the first token". Reading the clock after the open measures
  only the part of the wait that happens after the response headers.
- `arrived` goes between the chunk and the sink. What the client does with a chunk is
  the client's time. Reading the clock after the sink adds the consumer's cost to
  every inter-token gap and to TTFT.

**Chunks are not tokens.** A stream delivers messages; the metrics are defined on
tokens. A chunk with $k$ tokens is $k$ tokens that arrived together, so it contributes
$k$ timestamps (and $k - 1$ gaps of zero), and a chunk with none contributes nothing:
it cannot be the first token, and it cannot be the last. With this rule
`(t_n - t_1) / (n - 1)` is the true mean time per token whatever the server's
batching, and TTFT ignores the role-only chunk that OpenAI-compatible servers send
before any content.

**A failure is not a fast request.** The record of a failed or empty stream carries
`None` latencies, and `summarize` leaves such records out. The alternative, a zero or
the time of the failure, enters the percentiles as the fastest request of the run:
the more requests fail, the better the tail looks.

**Pooled gaps.** The ITL percentile is over every gap of every request. One mean per
request is TPOT, a different question: a stream that stalls for two seconds once and
is otherwise fast has a fine mean, and the stall is exactly what a p99 ITL is for.

**The closed loop** keeps `concurrency` requests in flight by replacing each one as
it finishes. Workers sharing a counter do it in a few lines; so does one task per
request behind an `asyncio.Semaphore`, whose waiters are woken in order.

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

`except OSError` covers what a transport can do to a request (refused, reset, timed
out: `ConnectionError` and `TimeoutError` are subclasses) and nothing else. A
`TypeError` from a bug in the benchmark must not be filed as a failed request, and
`asyncio.CancelledError` must pass through so that a canceled benchmark stops.

## Common mistakes

- **Starting the clock after the first await.** `stream = await open_stream()` and
  then `t_start = now()` reads naturally and drops the connection and queueing time
  from TTFT and E2E: 0.5 s instead of 0.75 s in the example.
- **Stamping after processing.** `await sink(chunk)` before `now()` shifts every
  timestamp by the consumer's delay. In a real client the same bug is parsing JSON,
  detokenizing, or printing before reading the clock.
- **Counting a failed stream as a success.** Catching the exception and falling
  through to the normal path yields a success with the tokens that happened to
  arrive, and with no tokens a success with zero latency.
- **TTFT at the first chunk.** The first chunk of a chat stream usually has a role
  and no content. It arrives when the server has accepted the request, long before
  the prefill is done.
- **One timestamp per chunk.** Then `n` is the number of chunks, and TPOT is the time
  per chunk: too large by the batching factor.
- **E2E at the end of the stream.** The usage chunk and the closing of the connection
  come after the last token; a user has their answer before that.
- **Nearest-rank instead of interpolation** (or the reverse). Both are legitimate
  definitions and they differ on small samples; a report has to use the one it
  states. The p50 of four TTFTs here is 0.375 by interpolation and 0.25 by nearest
  rank.
- **A percentile of per-request means for ITL.** That is the TPOT percentile again.
- **Failed records in the summary.** Entering `None` as 0 moves every percentile
  down.
- **Waves.** `gather` over groups of `concurrency` waits for the slowest request of
  each group before sending anything new: in the example request 2 is sent at 1.0
  instead of 0.5, and the server sees half the intended load.
- **Results in completion order.** Appending records as probes finish scrambles
  them; store by index.
- **The real clock.** `time.monotonic()` and the loop's clock agree in production
  and not under simulation; code that takes its time from the loop can be tested in
  microseconds.

## Complexity

`probe` is $O(\text{tokens})$ time and memory; `summarize` sorts, $O(G \log G)$ for
$G$ gaps; the closed loop adds $O(n)$.

## In the wild

- **vLLM's benchmark client** (`vllm/benchmarks/lib/endpoint_request_func.py`) reads
  `time.perf_counter()` for each message before parsing it, sets TTFT at the first
  chunk that carries output and skips the role-only opening chunk, and takes the
  latency at the last chunk with output, not at the usage chunk. It differs from this
  problem in one respect: it records one ITL per chunk, and gets the token count from
  the server's usage report, so its TPOT is `(latency - ttft) / (output_tokens - 1)`
  while its ITL list has one entry per message. Its percentiles are `numpy.percentile`
  with the default method, the interpolation used here.
- **Closed versus open loop.** A closed loop with `concurrency` workers is what
  "N concurrent users" means, and its offered rate is whatever the server's speed
  allows: when the server slows down, the client sends less, and the latencies it
  would have measured during the slowdown are never recorded (coordinated omission).
  Use it to measure throughput at a concurrency; use an open loop, which sends on a
  schedule regardless of responses, to measure latency at a rate.
- **A slow sink** is not hypothetical: a benchmark client written in Python that
  detokenizes and logs in the receive loop can spend longer per chunk than a fast
  server takes to produce it. Chunks then queue in the socket, arrival times bunch
  up, and the measured ITL describes the client. The fix is the same as here (stamp
  first) plus keeping the receive loop free of work.
