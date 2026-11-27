Trace a single sequence through the pipeline and write down when each stage is busy. How much of the time is each device idle? Now add a second sequence and ask what the earliest moment is that stage 0 can start on it, and what must finish before it can advance to each later stage.

---

Every service and transfer duration is known in advance, so you do not have to wait for tasks to finish to know when events happen. When a stage starts a job at time `T`, you immediately know when the stage becomes free (`T + service`) and what becomes ready next. If it is not the last stage, the job becomes ready at the next stage at `T + service + transfer`; at the last stage, the sequence's next token becomes ready at stage 0 at `T + service`. Store these future events in a min-heap ordered by time, and keep a per-stage min-heap of waiting jobs ordered by `(ready_time, request_index)`. The scheduler sleeps until the next event time, while each started job runs as its own task calling `await pipe.run(...)` followed by `await pipe.transfer(...)`.

---

Processing all events at a timestamp before making any scheduling decisions is what makes the timeline exact:

```python
now = calendar[0][0]
await asyncio.sleep(now - loop.time())
while calendar and calendar[0][0] == now:  # 1. everything that happens now
    ...  # mark a stage free, or push a job onto its stage's waiting heap
for stage in range(n_stages):  # 2. only then, the decisions
    if free[stage] and waiting[stage]:
        ...  # pop the job, start it as a task, push its future events
```

If you dispatch as each event arrives (for example, using one lock per stage that sequences race to acquire), a newly arriving request can claim stage 0 ahead of an older sequence's returning token at the same timestamp simply because its timer woke first.
