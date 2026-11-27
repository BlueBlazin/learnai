Start by imagining the simplest loop: for each request, sleep for the gap and then `await send(i)`. Now suppose the server takes 2 seconds per response and the gaps are 0.25 seconds. When is request 5 actually sent, when should it have been sent, and which `await` causes the delay?

---

The pacing loop must never await a server response. Instead, maintain `intended` as a running sum starting at `loop.time()` when `generate_load` is called:

```python
intended += -math.log(1.0 - uniforms[i]) / rate
```

For each request, sleep for `intended - loop.time()` (if positive), spawn a task with `loop.create_task(one(i, intended))`, and move on to the next request. After spawning all tasks, wait for them to finish and return the collected records.

---

Inside the per-request task:

```python
async def one(i, intended):
    if slots is not None:
        await slots.acquire()  # asyncio.Semaphore(max_in_flight): FIFO
    sent = loop.time()
    try:
        await send(i)
        ok = True
    except OSError:
        ok = False
    finally:
        if slots is not None:
            slots.release()
    done = loop.time()
    records[i] = {
        "intended_s": intended,
        "sent_s": sent,
        "done_s": done,
        "latency_s": done - intended,
        "ok": ok,
    }
```

Acquiring the semaphore inside `one` delays only that request, never the pacing schedule. Compute `"latency_s"` as `done - intended` rather than `done - sent`. Finally, await all tasks with `await asyncio.wait(tasks)` when `tasks` is non-empty.
