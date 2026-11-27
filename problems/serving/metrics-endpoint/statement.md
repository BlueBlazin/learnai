Implement an HTTP generation server with `POST /generate` and `GET /metrics` endpoints. The generation endpoint runs a language model, and the metrics endpoint exposes counters, a gauge, and a time-to-first-token histogram in the Prometheus text format.

![This problem is Metrics, the counters and histogram a scraper reads from the replica.](assets/map.svg)

## Specification

```python
def make_app(lm, clock=time.monotonic): ...
```

`make_app` returns an ASGI application. The starter builds it with [Starlette](https://www.starlette.io/), though a raw ASGI callable is equally acceptable. The tests serve the app on a local socket and talk to it over HTTP. Each app maintains its own metrics, all starting at zero.

**The model.** `lm` behaves like the kit's [`ScriptedLM`](../../../python/src/learnai/kit/scriptedlm.py):

- `lm.stream(prompt, max_tokens)` returns an async iterator that yields generated pieces (`str`), up to `max_tokens` pieces in total. One piece counts as one generated token. Iterating may raise `RuntimeError` at any point (before the first piece or after several) to indicate a failed generation. Calling `lm.stream(...)` itself does not raise.
- `lm.count_tokens(text)` returns the number of tokens in a prompt.

**The clock.** `clock()` returns the current time in seconds. Read time only through this function. The tests pass a custom clock so they can check histogram observations exactly.

**`POST /generate`.** A request body is valid if it is a JSON object with a string `"prompt"` and an optional integer `"max_tokens"` `>= 1`. The default value for `"max_tokens"` is 16, and booleans (`true` and `false`) are not valid integers. Any other keys are ignored.

| outcome | status | JSON body |
|---|---|---|
| Invalid body (model is not called) | 400 | `{"error": "<message>"}` |
| Model raised `RuntimeError` | 500 | `{"error": "<message>"}` |
| Generation completed | 200 | `{"text": <the pieces joined>, "prompt_tokens": <lm.count_tokens(prompt)>, "completion_tokens": <number of pieces>}` |

Error messages can be any non-empty string. Any other path returns status 404 and does not change any metric.

**`GET /metrics`.** This endpoint returns status 200 with content type `text/plain; version=0.0.4` (a `charset=utf-8` parameter is allowed) and exposes these five metrics:

| name | type | value |
|---|---|---|
| `vllm:num_requests_running` | gauge | Valid `/generate` requests whose generation has started and whose response has not been produced yet. |
| `vllm:prompt_tokens_total` | counter | Sum of `lm.count_tokens(prompt)` over all valid requests, including those that later fail. |
| `vllm:generation_tokens_total` | counter | Total number of pieces received from the model, including pieces from generations that later fail. |
| `http_requests_total{status="..."}` | counter | Number of `POST /generate` responses with that status. The three series `"200"`, `"400"`, and `"500"` are always present and start at 0 before their first event. |
| `vllm:time_to_first_token_seconds` | histogram | One observation per request that receives at least one piece, computed as `clock()` when the first piece arrives minus `clock()` when the handler began the request. |

The four `vllm:` names match vLLM's own metrics. vLLM reports HTTP statuses through its web framework outside the `vllm:` namespace, which `http_requests_total` stands in for here.

A request counts as running from before its first piece is requested until its response is ready, whether it succeeds or fails. When no request is in flight, the gauge is 0. A request that fails after receiving its first piece keeps its histogram observation.

**The histogram.** The time-to-first-token histogram uses the upper bounds `TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5)` seconds and exposes three kinds of series:

- `vllm:time_to_first_token_seconds_bucket{le="<bound>"}` for each bound in increasing order, followed by `le="+Inf"`. Values are **cumulative**, meaning each bucket counts all observations less than or equal to its bound. An observation equal to a bound belongs in that bound's bucket, and the `+Inf` bucket counts every observation.
- `vllm:time_to_first_token_seconds_sum` holds the sum of all observations.
- `vllm:time_to_first_token_seconds_count` holds the total number of observations, equal to the `+Inf` bucket.

**The text format.** Follow the subset of the [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) used here:

- Encode output as UTF-8 text. Every line, including the last, must end with a line feed (`\n`).
- Format each sample as `<name> <value>` or `<name>{<label>="<value>"} <value>`. Values must be numbers that `float()` can parse (such as `3`, `3.0`, or `1.78125`), and the final bucket bound must be written as `+Inf`.
- Place a `# TYPE <name> <type>` line (with type `counter`, `gauge`, or `histogram`) before each metric's first sample. For the histogram, use `<name>` = `vllm:time_to_first_token_seconds` without a suffix. A `# HELP <name> <text>` line is optional, and each metric may have at most one `# TYPE` and one `# HELP` line.
- Keep all lines of a metric together in one contiguous group without interleaving other metrics. Groups may appear in any order.
- Do not emit duplicate samples with the same name and labels. Other lines starting with `#` are treated as comments, and additional metrics are allowed.

## Example

Suppose the model takes 0.25 s to produce the first piece for `"the quick brown fox"`, 0.03125 s for `"hi"`, and 1.5 s for `"tell me a story"`, where it fails after yielding two pieces. Consider four sequential requests:

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

After these four requests, `GET /metrics` returns output matching the following (exact `# HELP` text is up to you):

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

The prompt token total is 4 + 1 + 4 = 9 because the failed request's prompt still counts. The generation token total is 3 + 1 + 2 = 6 because the failed request's two pieces still count. The three time-to-first-token observations are 0.25, 0.03125, and 1.5. The observation at 0.25 falls into the `le="0.25"` bucket, and every larger bucket includes all counts from smaller buckets.

## Constraints

- Tests send at most a few hundred requests, and prompts and pieces are short strings.
- Counters, the gauge, and bucket counts are compared exactly. `vllm:time_to_first_token_seconds_sum` is compared within `1e-9`.
- The test clock advances only while the model produces a piece, by exact binary floating-point amounts. Reading the start time anywhere between receiving the request and requesting the first piece gives the same result.
- Counters and the histogram are scraped only when no request is in flight, so you may update them either as pieces arrive or when a request finishes. The gauge is also scraped while two requests are blocked inside the model, so `/metrics` must respond while generations are running.
- The exposition is parsed strictly according to the format rules above. Any violation fails the test.
- Do not block the event loop (no `time.sleep` and no synchronous I/O).

## Why this matters

Production serving fleets rely on scraped metrics rather than manual log inspection. Every few seconds, a scraper pulls endpoints like `/metrics` so autoscalers can react to running and waiting gauges, alerts can fire on p99 time to first token, and teams can plan capacity from token counters.

The exposition conventions make those numbers composable across replicas. Cumulative bucket counters can be summed across servers and passed to `histogram_quantile`, while a missing `+Inf` bucket breaks quantile computation. Pre-initializing error series at zero distinguishes "no errors" from "not reporting." Failure paths matter most because they run least often. If a handler forgets to decrement a gauge on an exception, the count drifts upward with every error until the autoscaler mistakes leaked counts for real load.
