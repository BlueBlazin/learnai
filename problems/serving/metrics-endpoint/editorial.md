## The idea

**Three kinds of numbers.** A *counter* only goes up; consumers take its rate, so its
absolute value and a restart to zero do not matter. A *gauge* is a current level that
goes both ways. A *histogram* is a set of counters: one per bucket, plus the sum and
the count of the observations.

**Why buckets are cumulative.** The bucket `le="0.25"` counts the observations at or
below 0.25 s, which includes everything in `le="0.1"`. Three properties follow, and
they are the reason for the convention. Buckets are counters, so they can be rated and
summed across replicas like any other counter. Dropping a bucket loses resolution
without corrupting the others. And a quantile is found by walking the buckets until
the cumulative count passes the rank, which is what `histogram_quantile` does. The
`+Inf` bucket is the last step of that walk; the format requires it and requires it
to equal `_count`.

**Store plain counts, accumulate on output.** Internally, the simplest correct state
is one count per bucket, with the observation placed by `bisect_left` on the bounds:
the first bound greater than or equal to the value, so a value on a bound lands in
that bound's bucket. A running sum at render time makes the counts cumulative, and
the last one is `_count` by construction, so the two can never disagree.

**Metrics change on every exit.** The handler has three ways out (400, 500, 200), and
each metric must be right on all of them. The gauge is the fragile one: it is
incremented once and must be decremented exactly once, so the decrement belongs in
`finally`. The token counters deliberately include failed generations, because the
GPU did that work; a capacity plan made from tokens that only count successes
underestimates load exactly when things go wrong.

**Inject the clock.** The handler reads time through the `clock` argument. In
production that is `time.monotonic` (never the wall clock, which can jump); in a test
it is a function the test controls, and the histogram becomes an exact, checkable
quantity instead of a flaky one.

## Reference solution (Python)

```python
class Metrics:
    def __init__(self):
        self.running = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.requests = dict.fromkeys(STATUSES, 0)
        self.ttft_counts = [0] * (len(TTFT_BUCKETS) + 1)   # the last is +Inf
        self.ttft_sum = 0.0

    def observe_ttft(self, seconds):
        self.ttft_counts[bisect_left(TTFT_BUCKETS, seconds)] += 1
        self.ttft_sum += seconds

    def render(self):
        ttft = "vllm:time_to_first_token_seconds"
        cumulative = list(accumulate(self.ttft_counts))
        bounds = [*TTFT_BUCKETS, float("inf")]
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running {self.running}",
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total {self.prompt_tokens}",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total {self.generation_tokens}",
            "# TYPE http_requests_total counter",
            *(f'http_requests_total{{status="{s}"}} {n}' for s, n in self.requests.items()),
            f"# TYPE {ttft} histogram",
            *(f'{ttft}_bucket{{le="{number(b)}"}} {c}'
              for b, c in zip(bounds, cumulative, strict=True)),
            f"{ttft}_sum {number(self.ttft_sum)}",
            f"{ttft}_count {cumulative[-1]}",
        ]
        return "\n".join(lines) + "\n"


def make_app(lm, clock=time.monotonic):
    metrics = Metrics()

    async def generate(request):
        try:
            prompt, max_tokens = parse_request(await request.body())
        except BadRequest as exc:
            metrics.requests["400"] += 1
            return JSONResponse({"error": str(exc)}, status_code=400)

        started = clock()
        prompt_tokens = lm.count_tokens(prompt)
        metrics.prompt_tokens += prompt_tokens
        metrics.running += 1
        pieces = []
        stream = lm.stream(prompt, max_tokens)
        try:
            async for piece in stream:
                if not pieces:
                    metrics.observe_ttft(clock() - started)
                pieces.append(piece)
                metrics.generation_tokens += 1
        except RuntimeError as exc:
            metrics.requests["500"] += 1
            return JSONResponse({"error": f"generation failed: {exc}"}, status_code=500)
        finally:
            metrics.running -= 1
            await stream.aclose()
        metrics.requests["200"] += 1
        return JSONResponse({"text": "".join(pieces), "prompt_tokens": prompt_tokens,
                             "completion_tokens": len(pieces)})
    ...
```

`number` writes a float with `repr` and infinity as `+Inf`; `parse_request` is the
validation of the statement. The full file also has the `# HELP` lines.

No lock is needed: the handlers run on one event loop, and no metric update spans an
`await`. A multi-threaded or multi-process server needs a lock or per-process
aggregation here.

## Common mistakes

- **Buckets that are not cumulative.** Printing the per-bucket counts gives a
  plausible-looking histogram from which every quantile comes out wrong, usually far
  too low.
- **No `+Inf` bucket.** Observations above the largest bound vanish from the buckets
  while `_count` still includes them; `histogram_quantile` cannot work without it.
- **`<` for `<=`.** `le` is "less than or equal". With `bisect_right` (or a `<`
  comparison) a value exactly on a bound moves up one bucket.
- **Leaking the gauge on errors.** Decrementing after the loop instead of in
  `finally` leaves one phantom running request per failure. The graph looks like load
  that never ends.
- **Timing the whole generation.** Observing after the loop measures end-to-end
  latency under the name of time to first token.
- **Reading the real clock** when a clock is passed in. The code works and cannot be
  tested.
- **Counting tokens only on success.** The failed generation's prompt was processed
  and its pieces were generated.
- **Leaving 400s out of the request counter.** The error rate is the ratio of two
  series of the same counter; a status that is never counted is an error rate of zero.
- **Series that appear with their first event.** Until the first 500 there is no
  `status="500"` series, so `rate(...)` over it returns nothing instead of 0, and an
  alert on it never initializes. Create known label values at 0.
- **Format slips**: no `# TYPE` lines (everything becomes untyped), a body without the
  final line feed, the wrong content type, or returning the text through a JSON helper,
  which quotes it.

## Complexity

An observation is $O(\log B)$ for $B$ buckets; a scrape is $O(B)$ plus the number of
series. Memory is constant: it does not grow with the number of requests, which is the
point of a histogram compared with keeping samples.

## In the wild

- **vLLM** exports the metrics used here on `/metrics` of its OpenAI-compatible
  server, among them the gauges `vllm:num_requests_running` and
  `vllm:num_requests_waiting`, counters for prompt and generation tokens, and the
  histogram `vllm:time_to_first_token_seconds`; its documentation lists the counters
  without the `_total` suffix, which the Python client library appends in the
  exposition ([vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)).
  Real series also carry a `model_name` label.
- **The format** is specified in
  [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/):
  "A histogram must have a bucket with `{le="+Inf"}`. Its value must be identical to
  the value of `x_count`", the lines of a metric "must be provided as one single
  group", and "the last line must end with a line feed character". A colon in a
  metric name is legal; by convention it is reserved for recording rules, and vLLM
  uses it as a namespace anyway.
- **TGI** (`tgi_request_duration`, `tgi_queue_size`, ...), **Triton** (`nv_inference_*`),
  and **llama.cpp**'s server (`llamacpp:prompt_tokens_total`, with `--metrics`) expose
  the same kinds of series under their own names. In real code the registry and the
  rendering come from a client library (`prometheus_client` in Python), which also
  solves the multi-process case; writing it once by hand is how the conventions stop
  being magic.
- **Choosing buckets** is choosing where quantiles are accurate: the estimate
  interpolates within a bucket, so bounds should be dense around the objective (a
  500 ms TTFT objective wants bounds near 0.5). Native histograms, with exponential
  buckets chosen automatically, remove that decision in newer Prometheus versions.
