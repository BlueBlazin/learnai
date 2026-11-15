Keep two structures, a waiting queue and a running list, and decide for each what
order it is in and where a preempted sequence re-enters. Then plan a step in two
passes over them: first the sequences that already hold memory, then the ones that
want some. The engine tells you everything about memory; you only need to count
what you have promised in the step you are planning.

---

A decoding sequence needs a new block exactly when `engine.kv_tokens(id)` is a
multiple of the block size. Walk the running list from the oldest. When a block is
needed and `engine.free_blocks - promised` is 0, pop the *last* element of the
running list, `engine.preempt` it, and push it to the *left* of the waiting queue.
Because `preempt` frees blocks at once, `engine.free_blocks` already reflects it.
Note that the sequence you are serving can be the one you pop. A preempted request
is still live in the engine: remember its id, so that later you prefill
`engine.remaining_prefill(id)` instead of admitting it again.

---

```python
promised, evicted, decode = 0, False, []
for request in running[:]:
    if request not in running:
        break  # preempted, and so is everything after it
    if engine.kv_tokens(request.id) % block == 0:
        while engine.free_blocks - promised < 1:
            victim = running.pop()
            engine.preempt(victim.id)
            preempted.add(victim.id)
            waiting.appendleft(victim)
            evicted = True
            if victim is request:
                break
        if request not in running:
            break
        promised += 1
    decode.append(request.id)
```

Admission then loops while `not evicted`, there is a free slot, and the front of
`waiting` leaves `watermark_blocks` available (or 0, if `decode` and `prefill` are
both still empty).
