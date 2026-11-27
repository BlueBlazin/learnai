## The idea

**Three kinds of numbers.** A *counter* only increases. Consumers compute rates from it, so its absolute value and resets to zero do not matter. A *gauge* tracks a current level that moves both up and down. A *histogram* groups a set of bucket counters together with the sum and total count of observations.

**Why buckets are cumulative.** The bucket `le="0.25"` counts all observations at or below 0.25 s, including everything in `le="0.1"`. This convention gives Prometheus three useful properties. First, because buckets are counters, they can be rated and summed across replicas like any other counter. Second, dropping a bucket reduces resolution without corrupting the remaining buckets. Third, `histogram_quantile` estimates quantiles by walking the buckets until the cumulative count crosses the target rank. The `+Inf` bucket closes that walk, which is why the format requires it and requires it to equal `_count`.

**Store plain counts, accumulate on output.** The simplest internal representation stores one non-cumulative count per bucket, plus an extra slot for `+Inf`. Calling `bisect_left` on `TTFT_BUCKETS` returns the index of the first bound greater than or equal to the observation, so values on a bound land in that bound's bucket. Computing a running sum at render time produces the cumulative counts, and the final cumulative count equals `_count` by construction.

**Update metrics on every exit.** The `/generate` handler has three exit paths (400, 500, and 200), and each metric must stay accurate on all three. The running gauge is the easiest to break because it must be incremented once and decremented exactly once, so the decrement belongs in a `finally` block. The token counters intentionally include failed generations because the GPU still performed that work. Excluding failed requests would undercount load right when the server is under stress.

**Inject the clock.** Reading time through the `clock` argument keeps the handler testable. Production passes `time.monotonic` rather than wall-clock time, which can jump backward or forward. Tests pass a controlled clock so histogram observations can be verified exactly without flakiness.

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

Here `number` formats floats with `repr` and infinity as `+Inf`, and `parse_request` performs the JSON validation from the specification. The full reference file also includes `# HELP` lines.

No lock is needed because all handlers run on a single event loop and no metric update spans an `await`. A multi-threaded or multi-process server would require a lock or per-process aggregation.

## Common mistakes

- **Non-cumulative buckets.** Printing raw per-bucket counts produces a plausible-looking histogram whose quantiles come out wrong, usually far too low.
- **Missing `+Inf` bucket.** Observations above the largest finite bound disappear from the buckets while `_count` still includes them, breaking `histogram_quantile`.
- **Using `<` instead of `<=`.** Because `le` means "less than or equal," using `bisect_right` or `<` shifts observations that land exactly on a bound into the next bucket.
- **Leaking the gauge on errors.** Decrementing `running` after the loop instead of in `finally` leaves one phantom running request per failure, making the graph look like permanent load.
- **Timing the whole generation.** Recording the observation after the loop finishes measures end-to-end latency instead of time to first token.
- **Reading the real clock.** Ignoring the injected `clock` argument prevents deterministic testing of the histogram.
- **Counting tokens only on success.** A failed generation still processed its prompt and generated pieces before failing.
- **Omitting 400 responses from the request counter.** Error rates are ratios of series within the same counter, so uncounted error statuses hide failures.
- **Creating series only on their first event.** Without pre-initializing `status="500"` at 0, `rate(...)` returns no data until the first error and alerts may never initialize.
- **Format slips.** Common exposition bugs include omitting `# TYPE` lines, missing the trailing `\n`, sending the wrong content type, or returning the metrics string through a JSON helper that wraps it in quotes.

## Complexity

For $B$ buckets, recording an observation takes $O(\log B)$ time, and rendering a scrape takes $O(B)$ time plus the number of other series. Memory usage is $O(B)$, which stays constant regardless of how many requests are served.

## In the wild

- **vLLM** exports these metrics on `/metrics` in its OpenAI-compatible server, including `vllm:num_requests_running`, `vllm:num_requests_waiting`, prompt and generation token counters, and `vllm:time_to_first_token_seconds` ([vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)). Its documentation lists the counters without the `_total` suffix because the Python client library appends `_total` during exposition, and real deployments also attach a `model_name` label.
- **Prometheus exposition format** specifies these rules directly ([Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/)). A histogram must include a `{le="+Inf"}` bucket equal to `x_count`, all lines of a metric must appear in a single group, and the last line must end with a line feed. Colons in metric names are legal (conventionally reserved for recording rules, though vLLM uses them as a namespace).
- **Other servers** such as TGI (`tgi_request_duration`, `tgi_queue_size`), Triton (`nv_inference_*`), and llama.cpp (`llamacpp:prompt_tokens_total` with `--metrics`) expose similar series under their own names. Production Python services normally use `prometheus_client` for registration, rendering, and multi-process support, while implementing the format once by hand demystifies the conventions.
- **Bucket design** determines where quantile estimates are accurate because Prometheus interpolates within each bucket. Bounds should be dense around your latency target (for example, near 0.5 s for a 500 ms TTFT objective). Newer Prometheus versions also support native histograms with automatic exponential buckets.
