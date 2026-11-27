Maintain two collections: a waiting queue and a running list. Decide the ordering of each collection and where a preempted sequence should be reinserted. Plan every step in two passes, handling sequences that already hold memory before sequences waiting to be admitted. The engine tracks all committed block counts, so you only need to count the blocks you have promised during the current planning pass.

---

A decoding sequence needs a new block exactly when `engine.kv_tokens(id)` is a multiple of `engine.block_tokens`. Walk the running list from oldest to newest. Whenever a sequence needs a block and `engine.free_blocks - promised` is 0, pop the *last* element of the running list, call `engine.preempt` on it, and push it to the *left* of the waiting queue. Because `engine.preempt` frees blocks immediately, `engine.free_blocks` updates right away. Watch out for the case where the sequence you are currently serving is the one you pop. Remember the IDs of preempted requests because they stay live in the engine. When they resume, prefill `engine.remaining_prefill(id)` tokens instead of calling `engine.admit` again.

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

After this pass, loop over the front of `waiting` while `not evicted`, `len(running) < max_running`, and admitting the front request leaves at least `watermark_blocks` available (or `0` if both `decode` and `prefill` are still empty).
