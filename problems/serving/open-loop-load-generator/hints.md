Write the simplest loop first: for each request, wait for the gap, then `await
send(i)`. Now suppose the server takes two seconds to answer and the gaps are a
quarter of a second. When does request 5 go out? When should it? Which `await` in
your loop is the reason?

---

The pacing loop must never await a response. It computes the next intended time as
an absolute time, sleeps until then, starts the request as its own task
(`loop.create_task(one(i, intended))`), and goes on. Only after the last request has
been started does it wait for the tasks, and then returns the records.

Keep the intended time as a running sum, `intended += -math.log(1.0 - u) / rate`,
starting from `loop.time()` at the call, and sleep for `intended - loop.time()`. The
schedule is then fixed by the uniforms; nothing that happens later can move it.

---

Inside the task for one request:

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

The wait for a slot happens in the request's own task, after its intended time is
fixed, so it delays that request and no schedule. The latency subtracts `intended`,
not `sent`. Wait for the tasks with `asyncio.wait(tasks)` (skip it when there are
none).
