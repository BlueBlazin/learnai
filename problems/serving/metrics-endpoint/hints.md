Break the problem into two parts: keeping the five metrics accurate across successes, failures, and concurrent requests, and formatting them as Prometheus text. Start by writing down the exact moment each metric changes, and check which updates must happen on *every* exit path of the handler, including when the model raises `RuntimeError`.

---

Increment the running gauge before requesting the first piece, and decrement it inside a `finally` block so that normal completion and exceptions both clean up. Read `clock()` once at the start of the request and once when the first piece arrives (`if not pieces:` inside the loop), then record the difference. Increment `http_requests_total` on each of the three exit paths, and pre-initialize all three status keys at 0 (`dict.fromkeys(STATUSES, 0)`).

For the histogram, store one count per bucket plus one extra slot for `+Inf`, along with the running sum. Place each observation into the first bucket whose bound is greater than or equal to the observed value.

---

Compute cumulative bucket counts only when rendering:

```python
index = bisect.bisect_left(TTFT_BUCKETS, seconds)   # first bound >= seconds; len() means +Inf
self.ttft_counts[index] += 1

cumulative = list(itertools.accumulate(self.ttft_counts))
for bound, count in zip([*TTFT_BUCKETS, math.inf], cumulative):
    le = "+Inf" if bound == math.inf else repr(bound)
    lines.append(f'vllm:time_to_first_token_seconds_bucket{{le="{le}"}} {count}')
lines.append(f"vllm:time_to_first_token_seconds_sum {self.ttft_sum}")
lines.append(f"vllm:time_to_first_token_seconds_count {cumulative[-1]}")
```

Place each metric's `# TYPE` line directly before its samples, keep all lines for a metric together, and terminate the response with `"\n".join(lines) + "\n"`.
