There are exactly two places where the clock is read, and for each the question is
"before or after which await?". Everything that takes time in this problem is an
`await`: opening the stream, receiving a chunk, the sink. Decide for each timestamp
which of those waits belong to the server and must be inside the measured interval,
and which belong to the client and must stay outside it.

---

In `probe`: read `t_start` on the first line. Wrap the opening and the `async for` in
one `try`/`except OSError`, so a refusal and a broken stream end up in the same
place. Inside the loop read the clock first, append it once per token of the chunk
(`[arrived] * len(chunk)`), then `await sink(chunk)`. After the loop the status
follows from two facts: did an exception occur, and is `token_times` empty. Compute
the latencies only for `"ok"`, from `token_times` alone, as in `latency-metrics`.

---

For the closed loop, start `min(concurrency, n)` workers that share a counter; each
takes the next index, awaits a probe, stores the record at that index, and repeats
until the counter reaches `n`:

```python
async def worker():
    nonlocal next_index
    while next_index < n:
        i = next_index
        next_index += 1
        results[i] = await probe(partial(open_request, i), sink)


await asyncio.gather(*(worker() for _ in range(min(concurrency, n))))
```

Do not `gather` the requests in groups of `concurrency`: a group waits for its slowest
member. For `summarize`, pool the `"itl"` lists of the ok records into one list before
taking the percentile, and return `None` for an empty list.
