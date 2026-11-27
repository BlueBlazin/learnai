Solve the stages in order and maintain one invariant from stage 2 onward: whenever `sim_engine.step` is called, the allocator holds the exact blocks for every token each sequence will have after that step plus all remaining prompt tokens reserved at admission. If every state transition (admission, decode, finish, preemption) updates the allocator immediately, how can you check available memory without maintaining separate reservation counters?

---

Stage 1 is `paged-kv-copy-on-write` with an optional `n_tokens` parameter on `fork`. Stage 2 adds three allocator calls to `continuous-batching-scheduler`: `append(id, prompt_tokens)` at admission, `append(id, 1)` per decode, and `free(id)` for every finished sequence.

For stage 3, plan each step in three passes that share one `budget`: decodes (1 token each), admitted sequences with `remaining_prefill > 0`, and new admissions, using `min(remaining, threshold, budget)` for each chunk. For stage 4, keep both the waiting queue and the list of sequences holding memory ordered by `(priority, arrival_number)`. The decode pass walks the holding list from the front while popping victims from the back. Because you allocate eagerly, `allocator.free_blocks()` is always exact.

---

The decode-and-preemption pass of stage 4, with `holding` sorted by rank:

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

Admitting the head of the queue requires `ceil((shared + prefill) / block_size) - shared // block_size` blocks, where `shared` is `0` before stage 5 and `prefill` is `remaining_prefill(id)` for a preempted sequence. Compare `free_blocks() - needed` against `watermark_blocks`, or against `0` when no sequence holds memory.

For stage 5, for an incoming request `r` and each candidate donor `s`, compute `shared = block_size * (min(common_prefix, min(sim.kv_tokens(s.id), s.prompt_tokens), r.prompt_tokens - 1) // block_size)`. Pick the donor with the largest `shared` (tie-breaking by best rank). If `shared > 0`, call `fork(donor, r.id, shared)`, `admit(r, cached_tokens=shared)`, and `append(r.id, r.prompt_tokens - shared)`.
