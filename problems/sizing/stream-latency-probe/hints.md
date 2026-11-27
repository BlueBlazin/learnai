You only read the clock in two places, and the key question for each is whether it goes before or after an `await`. Everything that consumes time in this problem is an `await` (opening the stream, receiving a chunk, or running `sink`). Decide which waits belong to the server and must be inside the measured interval, and which belong to the client and must stay outside it.

---

In `probe`, read `t_start` on the first line. Wrap both `await open_stream()` and the `async for` loop in a single `try`/`except OSError` block so connection failures and mid-stream failures follow the same path. Inside the loop, read the clock first, append `[arrived] * len(chunk)` to `token_times`, and then call `await sink(chunk)`. After the loop, determine `"status"` from whether an `OSError` occurred and whether `token_times` is empty. Compute latencies only when `"status" == "ok"`, using `token_times` alone.

---

For `run_closed_loop`, launch `min(concurrency, n)` worker coroutines that share a counter. Each worker claims the next request index, awaits `probe`, writes the record to `results[i]`, and repeats until the counter reaches `n`:

```python
async def worker():
    nonlocal next_index
    while next_index < n:
        i = next_index
        next_index += 1
        results[i] = await probe(partial(open_request, i), sink)


await asyncio.gather(*(worker() for _ in range(min(concurrency, n))))
```

Do not call `asyncio.gather` on fixed chunks of size `concurrency`, because each batch would wait for its slowest request. For `summarize`, pool the `"itl"` lists from all `"ok"` records into one flat list before computing the percentile, and return `None` whenever a metric has no values.
