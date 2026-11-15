Follow one sequence through the pipeline and write down when each stage is busy
with it. For how much of the time is each device idle? Now add a second sequence:
what is the earliest moment stage 0 could start on it, and what, exactly, would
have to be true at each later stage before it can continue there?

---

Every duration is known in advance, so you never have to wait and see when
something ends: when a stage starts a job at time `T`, you already know that the
stage is free at `T + service`, that the job is ready at the next stage at
`T + service + transfer`, or (at the last stage) that the sequence's next token is
ready at stage 0 at `T + service`. Keep those future events in a heap ordered by
time, and keep, per stage, a heap of waiting jobs ordered by
`(ready time, request index)`. The scheduler sleeps until the next event time,
and each job runs as its own task: `await pipe.run(...)`, then
`await pipe.transfer(...)`.

---

The order of work at one instant is what makes the schedule exact:

```python
now = calendar[0][0]
await asyncio.sleep(now - loop.time())
while calendar and calendar[0][0] == now:  # 1. everything that happens now
    ...  # mark a stage free, or push a job onto its stage's waiting heap
for stage in range(n_stages):  # 2. only then, the decisions
    if free[stage] and waiting[stage]:
        ...  # pop the job, start it as a task, push its future events
```

If you decide as each event comes in (for example with one lock per stage that
sequences race for), a new request that arrives at the instant an older
sequence's token returns can take stage 0 first, only because its timer happened
to fire first.
