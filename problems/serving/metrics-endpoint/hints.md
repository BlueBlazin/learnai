There are two halves: keeping five numbers correct while requests succeed, fail, and
overlap, and printing them. For the first half, write down for each metric the exact
moment it changes. Which of those moments must happen on *every* way out of the
handler, including the one where the model raises?

---

Increment the gauge before the first piece is requested and decrement it in a
`finally` block, so that success, a `RuntimeError`, and anything else all pass
through it. Read `clock()` once when the request starts and once when the first piece
arrives (`if not pieces:` inside the loop), and observe the difference. Count the
status in each of the three exits. Create the three status series at 0 up front
(`dict.fromkeys(STATUSES, 0)`).

For the histogram, store one count per bucket plus one for `+Inf`, and the sum. An
observation goes to the first bucket whose bound is greater than or equal to it.

---

Make the bucket counts cumulative only when rendering:

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

Put each metric's `# TYPE` line directly before its samples, keep each metric's lines
together, and finish with `"\n".join(lines) + "\n"`.
