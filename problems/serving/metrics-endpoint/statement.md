Build a small generation server that reports on itself: `POST /generate` runs the
model, and `GET /metrics` exposes counters, a gauge, and a time-to-first-token
histogram in the Prometheus text format. The format is a few lines of text; the work
is in the conventions a scraper relies on (cumulative buckets, `+Inf`, series that
exist before their first event) and in keeping the numbers right when requests fail.

![This problem is Metrics, the counters and histogram a scraper reads from the replica.](assets/map.svg)

## Specification

```python
def make_app(lm, clock=time.monotonic): ...
```

`make_app` returns an ASGI application. The starter builds it with
[Starlette](https://www.starlette.io/); a raw ASGI callable is equally acceptable. The
tests serve the app on a local socket and talk to it over HTTP. Each app has its own
metrics, all starting at zero.

**The model.** `lm` behaves like the kit's
[`ScriptedLM`](../../../python/src/learnai/kit/scriptedlm.py):

- `lm.stream(prompt, max_tokens)` returns an async iterator that yields the generated
  pieces (`str`), at most `max_tokens` of them. One piece is one generated token.
  Iterating may raise `RuntimeError` at any point, before the first piece or after
  several: the generation has failed. The call `lm.stream(...)` itself does not raise.
- `lm.count_tokens(text)` returns the number of tokens of a prompt.

**The clock.** `clock()` returns the current time in seconds. Read time only through
it. The tests pass a clock of their own, which is what makes the histogram checkable
exactly.

**`POST /generate`.** The body is valid if it is a JSON object with a string
`"prompt"` and, optionally, an integer `"max_tokens"` `>= 1` (default 16; `true` and
`false` are not integers). Other keys are ignored.

| outcome | status | JSON body |
|---|---|---|
| the body is invalid; the model is not called | 400 | `{"error": "<message>"}` |
| the model raised `RuntimeError` | 500 | `{"error": "<message>"}` |
| the generation completed | 200 | `{"text": <the pieces joined>, "prompt_tokens": <lm.count_tokens(prompt)>, "completion_tokens": <number of pieces>}` |

Messages are non-empty strings of your choice. Any other path gets status 404 and
changes no metric.

**`GET /metrics`** returns status 200, content type `text/plain; version=0.0.4`
(a `charset=utf-8` parameter is allowed), and these five metrics:

| name | type | value |
|---|---|---|
| `vllm:num_requests_running` | gauge | valid `/generate` requests whose generation has started and whose response has not been produced yet |
| `vllm:prompt_tokens_total` | counter | sum of `lm.count_tokens(prompt)` over all valid requests, including those that then fail |
| `vllm:generation_tokens_total` | counter | number of pieces received from the model, including the pieces of generations that then fail |
| `http_requests_total{status="..."}` | counter | number of `POST /generate` responses with that status; the three series `"200"`, `"400"`, and `"500"` are always present, at 0 before their first event |
| `vllm:time_to_first_token_seconds` | histogram | one observation per request that received at least one piece: `clock()` when the first piece arrives minus `clock()` when the handler began the request |

The four `vllm:` names are those of vLLM's own metrics. vLLM reports HTTP statuses
through its web framework's instrumentation, outside the `vllm:` namespace, which
`http_requests_total` stands in for here.

A request counts as running from before its first piece is requested until its
response is ready, whether it succeeds or fails; when no request is in flight the
gauge is 0. A request that fails after its first piece keeps its histogram
observation.

**The histogram** has the upper bounds `TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0,
2.5)` seconds and is exposed as:

- `vllm:time_to_first_token_seconds_bucket{le="<bound>"}` for each bound in
  increasing order, then `le="+Inf"`. The value is **cumulative**: the number of
  observations less than or equal to the bound. An observation equal to a bound counts
  in that bound's bucket. The `+Inf` bucket therefore counts every observation.
- `vllm:time_to_first_token_seconds_sum`: the sum of all observations.
- `vllm:time_to_first_token_seconds_count`: their number, equal to the `+Inf` bucket.

**The text format**
([Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/)),
as far as this problem uses it:

- UTF-8 text; every line, including the last, ends with a line feed (`\n`).
- A sample is `<name> <value>` or `<name>{<label>="<value>"} <value>`. Values are
  numbers as `float()` reads them (`3`, `3.0`, `1.78125`); the bound of the last
  bucket is written `+Inf`.
- Each metric has a line `# TYPE <name> <type>` with type `counter`, `gauge`, or
  `histogram`, before its first sample. For the histogram, `<name>` is
  `vllm:time_to_first_token_seconds`, without suffix. A line `# HELP <name> <text>`
  is optional. At most one of each per metric.
- All lines of one metric form one group: they are not interleaved with another
  metric's lines. The order of the groups is free.
- No two samples have the same name and labels. Other lines starting with `#` are
  comments. Additional metrics are allowed.

## Example

The model takes 0.25 s to produce the first piece for `"the quick brown fox"`,
0.03125 s for `"hi"`, and 1.5 s for `"tell me a story"`, for which it fails after two
pieces. Four requests, one after another:

```
POST /generate {"prompt": "the quick brown fox", "max_tokens": 3}
  -> 200 {"text": "Hello, world", "prompt_tokens": 4, "completion_tokens": 3}
POST /generate {"prompt": "hi"}
  -> 200 {"text": "Hey", "prompt_tokens": 1, "completion_tokens": 1}
POST /generate {"prompt": 7}
  -> 400 {"error": "'prompt' is required and must be a string"}
POST /generate {"prompt": "tell me a story"}
  -> 500 {"error": "generation failed: the model failed"}
```

Then `GET /metrics` returns (the reference's output; `# HELP` texts are free):

```
# HELP vllm:num_requests_running Number of requests currently being generated.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 0
# HELP vllm:prompt_tokens_total Number of prompt tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 9
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 6
# HELP http_requests_total Responses of POST /generate by status code.
# TYPE http_requests_total counter
http_requests_total{status="200"} 2
http_requests_total{status="400"} 1
http_requests_total{status="500"} 1
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.05"} 1
vllm:time_to_first_token_seconds_bucket{le="0.1"} 1
vllm:time_to_first_token_seconds_bucket{le="0.25"} 2
vllm:time_to_first_token_seconds_bucket{le="0.5"} 2
vllm:time_to_first_token_seconds_bucket{le="1.0"} 2
vllm:time_to_first_token_seconds_bucket{le="2.5"} 3
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 3
vllm:time_to_first_token_seconds_sum 1.78125
vllm:time_to_first_token_seconds_count 3
```

The prompt tokens are 4 + 1 + 4 (the failed request's prompt counts), the generation
tokens 3 + 1 + 2 (so do its two pieces). The three observations are 0.25, 0.03125,
and 1.5: the one at exactly 0.25 is in the `le="0.25"` bucket, and every bucket also
contains everything the smaller buckets contain.

## Constraints

- At most a few hundred requests per test; prompts and pieces are short strings.
- Counters, the gauge, and bucket counts are compared exactly;
  `vllm:time_to_first_token_seconds_sum` within `1e-9`. The test clock advances only
  while the model produces a piece, by amounts that are exact in binary, so it does
  not matter where between receiving the request and asking for the first piece you
  read the start time.
- The counters and the histogram are read only while no request is in flight, so you
  may update them as pieces arrive or when the request ends. The gauge is also read
  while two requests are blocked inside the model: `/metrics` must answer while
  generations are running.
- The exposition is parsed strictly by the rules above; a violation fails the case.
- Do not block the event loop (no `time.sleep`, no synchronous I/O).

## Why this matters

Nobody watches a serving fleet by reading logs. Autoscalers add replicas when the
running and waiting gauges climb, alerts fire on the p99 of the time-to-first-token
histogram, and capacity plans are made from token counters, all scraped from an
endpoint like this one every few seconds. The conventions are what make the numbers
combine: buckets are cumulative counters so that they can be summed across replicas
and turned into quantiles by `histogram_quantile`, a missing `+Inf` bucket breaks
that function, and a series that does not exist until its first error makes
"no errors" indistinguishable from "not reporting". The failure paths matter most,
because they are exercised least: a gauge that is not decremented when a request
fails drifts upward with every error, and the autoscaler reads it as load.
