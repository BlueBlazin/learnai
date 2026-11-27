Implement the client-side measurement layer of a streaming LLM benchmark. Write `probe` to measure a single streamed request, `summarize` to aggregate latency percentiles across records, and `run_closed_loop` to run requests at fixed concurrency.

## Specification

```python
async def probe(open_stream, sink) -> dict: ...
def summarize(results: list[dict], q: float) -> dict: ...
async def run_closed_loop(open_request, sink, concurrency: int, n: int) -> list[dict]: ...
```

Read timestamps from the event loop's clock via `asyncio.get_running_loop().time()`, which returns seconds. The starter file wraps this call as `now()`. Under the judge, the event loop clock is simulated. Do not call `time.time`, `time.monotonic`, `time.perf_counter`, or `time.sleep`.

### `probe(open_stream, sink)`

- `open_stream` is a zero-argument async callable. Awaiting `open_stream()` sends the request and returns an async iterator of chunks once the response begins. Connecting, queueing at the server, and waiting for the response to start all happen inside this await and take time.
- A **chunk** is a list of token IDs. A chunk may hold several tokens when a server batches tokens under load, or zero tokens for metadata messages such as a role-only opening chunk, a keep-alive, or a closing usage chunk. The stream ends when the async iterator finishes.
- `sink` is an async callable that consumes the stream (for example, to render, detokenize, or log chunks). Pass every chunk to `await sink(chunk)` in arrival order, including empty chunks. Request the next chunk only after `sink` returns. `sink` takes time and never raises.

**Timestamps.**

- Read `t_start` when `probe` is called, before awaiting `open_stream()`.
- Read a chunk's arrival time immediately after the iterator returns it, before passing the chunk to `sink`.
- Assign that arrival time to every token in the chunk. A chunk with $k$ tokens adds $k$ equal entries to `token_times`.

**Outcome.** A transport failure is a recorded result rather than a crash. Catch `OSError` and its subclasses (such as `ConnectionError`, `ConnectionResetError`, and `TimeoutError`), and let all other exceptions propagate. Set `"status"` as follows:

| status | when |
|---|---|
| `"ok"` | the stream ended normally and carried at least one token |
| `"empty"` | the stream ended normally without a single token |
| `"error"` | `open_stream()` or stream iteration raised `OSError` or a subclass |

**Return record.** Return a dict with exactly the seven keys below. They report time to first token (`"ttft"`), inter-token latencies (`"itl"`), time per output token (`"tpot"`), and end-to-end latency (`"e2e"`). Let $t_1 \le \dots \le t_n$ denote the $n$ timestamps in `token_times`.

| key | value for status `"ok"` | otherwise |
|---|---|---|
| `"status"` | `"ok"` | `"empty"` or `"error"` |
| `"t_start"` | float | float |
| `"token_times"` | list of $n$ floats | tokens received before the end or failure (possibly `[]`) |
| `"ttft"` | $t_1 - t_\text{start}$ | `None` |
| `"itl"` | the $n - 1$ gaps $t_{i+1} - t_i$ | `[]` |
| `"tpot"` | $(t_n - t_1) / (n - 1)$, or `None` when $n = 1$ | `None` |
| `"e2e"` | $t_n - t_\text{start}$ | `None` |

Measure TTFT at the first **token**, not the first chunk. Measure E2E at the last **token**, not the end of the stream. Empty chunks before, between, or after tokens do not change either metric. A failed or empty request has no latency metrics, though any tokens received before failure remain in `token_times` for inspection.

### `summarize(results, q)`

`results` is a list of records returned by `probe`, and `0 <= q <= 100`. Only records with `"status" == "ok"` contribute latencies. Do not modify `results`. Return a dict with exactly these keys:

| key | value |
|---|---|
| `"requests"` | `len(results)` |
| `"ok"` | number of `"ok"` records |
| `"ttft"` | `q`-th percentile of `"ttft"` over `"ok"` records |
| `"tpot"` | `q`-th percentile of `"tpot"` over `"ok"` records whose `"tpot"` is not `None` |
| `"itl"` | `q`-th percentile of **all gaps pooled** across every `"ok"` record's `"itl"` list |
| `"e2e"` | `q`-th percentile of `"e2e"` over `"ok"` records |

If there are no values for a metric, its percentile is `None`. Compute percentiles using **linear interpolation**, matching the default method of `numpy.percentile`. Given $m$ sorted values $x_0 \le \dots \le x_{m-1}$, compute the fractional index $h$ and interpolate between its surrounding entries:

$$
h = (m - 1) \cdot \frac{q}{100}, \qquad i = \lfloor h \rfloor, \qquad
\text{result} = x_i + (h - i)\,(x_{i+1} - x_i).
$$

When $i = m - 1$, return $x_{m-1}$.

### `run_closed_loop(open_request, sink, concurrency, n)`

`open_request(i)` is an async callable that sends request `i` and returns its stream, just like `open_stream`. Probe requests `0, 1, ..., n - 1` with at most `concurrency` requests in flight at once:

- Start requests `0` through `min(concurrency, n) - 1` immediately when `run_closed_loop` is called.
- Whenever a probe returns (regardless of its status), immediately start the lowest-numbered request that has not yet been sent.
- Once every probe finishes, return the list of `n` records in request order so `result[i]` belongs to request `i`.

Send each request exactly once. Its `t_start` is the time it was sent.

## Example

Suppose a request is sent at $t = 2.0$. The response begins 0.25 s later and delivers chunks at these offsets from `t_start`:

| offset (s) | chunk | notes |
|---|---|---|
| 0.5 | `[]` | role-only opening chunk |
| 0.75 | 2 tokens | |
| 0.875 | 1 token | |
| 1.0 | 3 tokens | |
| 1.0625 | `[]` | closing usage chunk |
| 1.125 | | end of stream |

`probe` returns:

```python
{
    "status": "ok",
    "t_start": 2.0,
    "token_times": [2.75, 2.75, 2.875, 3.0, 3.0, 3.0],
    "ttft": 0.75,  # first token, not the empty chunk at 2.5
    "itl": [0.0, 0.125, 0.125, 0.0, 0.0],
    "tpot": 0.05,  # (3.0 - 2.75) / 5
    "e2e": 1.0,  # last token, not the stream end at 3.125
}
```

The record is identical whether `sink` returns immediately or takes 1/32 s per chunk.

Next, consider six records consisting of one `"error"`, one `"empty"`, and four `"ok"` records, where the four `"ok"` records have:
- `ttft`: `0.25`, `0.5`, `0.125`, `1.0`
- `tpot`: `0.125`, `0.25`, `None`, `0.0625`
- `itl`: `[0.125, 0.125]`, `[0.0, 0.25, 0.5]`, `[]`, `[0.0625]`
- `e2e`: `0.5`, `1.25`, `0.125`, `1.0625`

`summarize` produces:

```python
summarize(results, 50)
# {"requests": 6, "ok": 4, "ttft": 0.375, "tpot": 0.125, "itl": 0.125, "e2e": 0.78125}
summarize(results, 90)
# {"requests": 6, "ok": 4, "ttft": 0.85, "tpot": 0.225, "itl": 0.375, "e2e": 1.19375}
```

For p90 ITL, pooling all six gaps gives `[0.0, 0.0625, 0.125, 0.125, 0.25, 0.5]`. That yields $h = 5 \cdot 0.9 = 4.5$, which is halfway between `0.25` and `0.5`, giving `0.375`.

Finally, consider a closed loop with `concurrency = 2` and five requests lasting `1.0`, `0.5`, `0.25`, `1.0`, and `0.5` s. Requests 0 and 1 start at $t = 0$. Request 1 finishes at $t = 0.5$ and frees its slot for request 2. Request 2 finishes at $t = 0.75$ and frees that slot for request 3. Request 0 finishes at $t = 1.0$ and frees its slot for request 4. The five send times are therefore `0, 0, 0.5, 0.75, 1.0`.

## Constraints

- Streams have at most 200 chunks of at most 64 tokens each.
- `0 <= n <= 1000` and `1 <= concurrency <= 1000`.
- All timestamps in tests are multiples of 1/1024 s, so differences are exact. Floats are compared with an absolute tolerance of `1e-9`, and all other fields are compared exactly.
- `summarize` receives at most 100,000 gaps in total. `q` has at most one decimal place.
- If chunks from two different requests arrive at the same instant, they may reach `sink` in either order. Nothing else in the output depends on the ordering of simultaneous events.

## Why this matters

Every latency number in a serving benchmark comes from client-side measurement code like this, and measurement bugs rarely cause visible errors. Reading the clock after the connection opens hides server queueing from TTFT right when the server is overloaded. Reading the clock after processing each chunk blames a slow client on the model. Treating chunks as tokens distorts per-token speed whenever a server batches multiple tokens per message. Including failed streams as zero-latency successes makes tail percentiles look better as error rates climb.

A closed-loop load generator also has a well-known limitation. Because it only sends a new request when an earlier one finishes, its offered rate drops whenever the server slows down. That makes it good for measuring throughput at fixed concurrency, while an open-loop generator is needed to measure latency at a fixed arrival rate.
