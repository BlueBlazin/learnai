Write the measuring half of a benchmark client: send a request, consume its stream of
chunks, and produce the timestamps and latency metrics of that one request; then
summarize many such records and drive a closed loop of them. The arithmetic is that of
`latency-metrics`. What is new is where the numbers come from: a clock read at the
wrong moment, or a chunk mistaken for a token, gives a record that looks plausible and
measures something else.

## Specification

```python
async def probe(open_stream, sink) -> dict: ...
def summarize(results: list[dict], q: float) -> dict: ...
async def run_closed_loop(open_request, sink, concurrency: int, n: int) -> list[dict]: ...
```

Time is the event loop's clock, `asyncio.get_running_loop().time()`, in seconds; the
starter wraps it as `now()`. Under the judge that clock is simulated. Do not use the
real clock (`time.time`, `time.monotonic`, `time.perf_counter`, `time.sleep`).

### `probe(open_stream, sink)`

- `open_stream` is an async callable without arguments. `await open_stream()` sends
  the request and, once the response has begun, returns the stream: an async iterator
  of chunks. Connecting, queueing at the server, and waiting for the response to begin
  all happen inside that await and take time.
- A **chunk** is a list of token ids. It may hold several tokens (servers batch tokens
  into one message under load) or none (the opening chunk that only carries the role,
  a keep-alive, the closing chunk with usage). The stream ends when the iterator is
  exhausted.
- `sink` is an async callable, the consumer of the stream (it renders, detokenizes,
  logs). `probe` passes every chunk to it, `await sink(chunk)`, in arrival order,
  empty chunks included, and requests the next chunk only after `sink` has returned.
  `sink` takes time and never raises.

**Timestamps.**

- `t_start` is read when `probe` is called, before `open_stream` is awaited.
- A chunk's arrival time is read as soon as the iterator returns the chunk, before the
  chunk is passed to `sink`.
- Every token of a chunk gets the chunk's arrival time: a chunk of $k$ tokens
  contributes $k$ equal entries to `token_times`.

**Outcome.** A transport failure is a result, not a crash: `probe` catches `OSError`
and nothing else (an exception of another type is a bug in the caller's code and must
propagate). The record's `"status"` is

| status | when |
|---|---|
| `"ok"` | the stream ended normally and carried at least one token |
| `"empty"` | the stream ended normally without a single token |
| `"error"` | `open_stream()` or the iteration raised `OSError` or one of its subclasses (`ConnectionError`, `ConnectionResetError`, `TimeoutError`, ...) |

**The record** is a dict with exactly these keys. Write $t_1 \le \dots \le t_n$ for
`token_times`.

| key | value for status `"ok"` | otherwise |
|---|---|---|
| `"status"` | `"ok"` | `"empty"` or `"error"` |
| `"t_start"` | float | float |
| `"token_times"` | list of $n$ floats | the tokens received before the end or the failure (possibly `[]`) |
| `"ttft"` | $t_1 - t_\text{start}$ | `None` |
| `"itl"` | the $n - 1$ gaps $t_{i+1} - t_i$ | `[]` |
| `"tpot"` | $(t_n - t_1) / (n - 1)$, or `None` when $n = 1$ | `None` |
| `"e2e"` | $t_n - t_\text{start}$ | `None` |

TTFT is measured at the first **token**, not the first chunk, and the end-to-end
latency at the last token, not at the end of the stream: empty chunks before, between,
or after the tokens move nothing. A request that failed or produced nothing has no
latency; the tokens it did deliver are kept for inspection only.

### `summarize(results, q)`

`results` is a list of records and `0 <= q <= 100`. Only records with status `"ok"`
contribute latencies. The function returns a dict with exactly these keys:

| key | value |
|---|---|
| `"requests"` | `len(results)` |
| `"ok"` | the number of `"ok"` records |
| `"ttft"` | the `q`-th percentile of `"ttft"` over the ok records |
| `"tpot"` | the `q`-th percentile of `"tpot"` over the ok records whose `"tpot"` is not `None` |
| `"itl"` | the `q`-th percentile of **all gaps pooled**: every element of every ok record's `"itl"` |
| `"e2e"` | the `q`-th percentile of `"e2e"` over the ok records |

A percentile over no values is `None`. The percentile is **linear interpolation**, the
default of `numpy.percentile`: with the $m$ values sorted as $x_0 \le \dots \le
x_{m-1}$,

$$
h = (m - 1) \cdot \frac{q}{100}, \qquad i = \lfloor h \rfloor, \qquad
\text{result} = x_i + (h - i)\,(x_{i+1} - x_i),
$$

and $x_{m-1}$ when $i = m - 1$. It does not modify `results`.

### `run_closed_loop(open_request, sink, concurrency, n)`

`open_request(i)` is an async callable that sends request `i` and returns its stream,
like `open_stream`. The function probes requests `0, 1, ..., n - 1` with at most
`concurrency` of them in flight:

- when it is called, requests `0` to `min(concurrency, n) - 1` are sent at once;
- whenever a probe returns, whatever its status, the lowest-numbered request not yet
  sent is sent at that same instant;
- it returns, when every probe has returned, the list of the `n` records in request
  order (`result[i]` belongs to request `i`).

Each request is sent exactly once, and its `t_start` is the time it was sent.

## Example

A request is sent at $t = 2.0$. The response begins 0.25 s later and delivers, at
these offsets from the send time:

| offset | chunk | |
|---|---|---|
| 0.5 | `[]` | the role-only opening chunk |
| 0.75 | 2 tokens | |
| 0.875 | 1 token | |
| 1.0 | 3 tokens | |
| 1.0625 | `[]` | the usage chunk |
| 1.125 | | end of stream |

```python
{
    "status": "ok",
    "t_start": 2.0,
    "token_times": [2.75, 2.75, 2.875, 3.0, 3.0, 3.0],
    "ttft": 0.75,  # the first token, not the chunk at 2.5
    "itl": [0.0, 0.125, 0.125, 0.0, 0.0],
    "tpot": 0.05,  # (3.0 - 2.75) / 5
    "e2e": 1.0,  # the last token, not the end at 3.125
}
```

The record is the same whether `sink` returns at once or takes 1/32 s per chunk.

Four ok records with `ttft` 0.25, 0.5, 0.125, 1.0; `tpot` 0.125, 0.25, `None`, 0.0625;
`itl` `[0.125, 0.125]`, `[0.0, 0.25, 0.5]`, `[]`, `[0.0625]`; and `e2e` 0.5, 1.25,
0.125, 1.0625, together with one `"error"` and one `"empty"` record, summarize as

```python
summarize(results, 50)
# {"requests": 6, "ok": 4, "ttft": 0.375, "tpot": 0.125, "itl": 0.125, "e2e": 0.78125}
summarize(results, 90)
# {"requests": 6, "ok": 4, "ttft": 0.85, "tpot": 0.225, "itl": 0.375, "e2e": 1.19375}
```

The p90 of ITL is taken over the six pooled gaps `[0, 0.0625, 0.125, 0.125, 0.25,
0.5]`: $h = 4.5$, halfway between 0.25 and 0.5.

Closed loop: five requests that last 1.0, 0.5, 0.25, 1.0, and 0.5 s, two at a time,
are sent at $t = 0, 0, 0.5, 0.75, 1.0$. Request 2 takes the slot request 1 frees at
0.5, request 3 the same slot at 0.75, and request 4 the slot request 0 frees at 1.0.

## Constraints

- Streams have at most 200 chunks of at most 64 tokens; `0 <= n <= 1000`,
  `1 <= concurrency <= 1000`.
- All times in the tests are multiples of 1/1024 s, so every difference is exact;
  floats are compared with an absolute tolerance of `1e-9`, everything else exactly.
- `summarize` sees at most 100,000 gaps in total; `q` has at most one decimal place.
- Simultaneous events: two chunks of different requests that arrive at the same
  instant may reach `sink` in either order; nothing else in the results depends on
  the order of simultaneous events.

## Why this matters

Every latency number in a benchmark report was produced by code like this, and its
mistakes are invisible in the output. Start the clock after the connection is
established and the server's queueing time disappears from TTFT, exactly when the
server is overloaded and queueing is the story. Stamp a chunk after processing it and
a slow client is reported as a slow model. Count chunks as tokens and a server that
batches three tokens per message looks three times slower per token than it is, or
three times faster in tokens per second if the count comes from elsewhere. Count a
failed stream as a success with zero latency and the percentiles improve as the server
gets worse. The closed loop has its own trap: it is the simplest load generator, and
it only sends as fast as the server answers, which is what an open-loop generator
exists to fix.
