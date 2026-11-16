Solve the stages in order and keep one invariant in view from stage 2 on: when
`sim_engine.step` is called, the allocator holds exactly the tokens every admitted
sequence will have after that step, plus the prompt tokens it was promised at
admission. If every place that changes who holds memory (admission, a decode, a
finish, a preemption) makes the matching allocator call at that moment, what does
"how many blocks are available" become?

---

Stage 1 is `paged-kv-copy-on-write` with one more argument. Stage 2 is the loop of
`continuous-batching-scheduler` with three allocator calls: `append(id, prompt)` at
admission, `append(id, 1)` per decode, `free(id)` per finished id. For stage 3,
plan in three passes that share one `budget`: decodes (one token each), then
admitted sequences with `remaining_prefill > 0`, then admissions; every chunk is
`min(remaining, threshold, budget)`. For stage 4, keep both the waiting queue and
the list of sequences that hold memory sorted by `(priority, arrival number)`:
then the victim is always the last element of the second list, and the first pass
walks it from the front while popping victims from the back. Because you append
eagerly, `allocator.free_blocks()` is already the number of available blocks.

---

The first pass of stage 4, with `holding` sorted by rank:

```python
i, evicted, decode = 0, False, []
while i < len(holding):
    request = holding[i]
    i += 1
    if sim.remaining_prefill(request.id):
        continue
    if sim.kv_tokens(request.id) % block_size == 0:
        while allocator.free_blocks() < 1 and len(holding) >= i:
            preempt(holding.pop())  # allocator.free, sim.preempt, back to the queue
            evicted = True
        if len(holding) < i:
            break  # the one that asked was the victim
    allocator.append(request.id, 1)
    decode.append(request.id)
```

Admission of the head needs `ceil((shared + prefill) / block_size) - shared //
block_size` blocks, where `shared` is 0 until stage 5 and `prefill` is
`remaining_prefill(id)` for a preempted sequence; compare `free_blocks() - needed`
with `watermark_blocks`, or with 0 when nothing holds memory. For stage 5,
`shared = block_size * max over holders of min(common prefix, min(kv_tokens,
prompt_tokens), r.prompt_tokens - 1) // block_size`, then `fork(donor, r.id,
shared)`, `admit(r, cached_tokens=shared)`, `append(r.id, prompt_tokens - shared)`.
