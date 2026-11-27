## The idea

The key design decision in this capstone is to keep the allocator as the single source of truth for memory and update it **eagerly**. Whenever the planner decides that a sequence needs more blocks (at admission or before a decode), it calls the allocator immediately before running `sim_engine.step`. That way, the engine never tracks separate reservation counters or promised blocks. Available memory is always `allocator.free_blocks()`.

![The planner builds each step from the queue and the allocator's blocks, and the kit's `SimEngine` runs it.](assets/architecture.svg)

**Stage 1** implements the allocator from `paged-kv-copy-on-write` using a min-heap of free block ids, a reference count array, and per-sequence block tables and token counts. `append` computes the exact block count it needs (one copy block if a partially filled last block is shared, plus any growth blocks), verifies that `free_blocks()` is large enough *before mutating any state*, and only then pops from the heap. `free` decrements each block's reference count and pushes blocks back onto the heap when their count hits zero. The partial `fork` of stage 5 runs the same reference-increment loop over a prefix slice of the parent's table.

**Stage 2** connects the allocator to `continuous-batching-scheduler` at three points: `append(id, prompt_tokens)` at admission, `append(id, 1)` for each decoding sequence, and `free(id)` for every sequence in `result.finished`. Allocating the full prompt's blocks at admission ensures that later stages can prefill a prompt in chunks without ever stalling for memory mid-prefill.

**Stage 3** runs the `chunked-prefill-scheduling` planner in three passes sharing one token budget: decodes, already-admitted prompts with remaining prefill, and new admissions. Each chunk size is `min(remaining, threshold, budget)`. Scheduling decodes first guarantees that every generating sequence produces a token on every step while the token budget caps step duration.

**Stage 4** unifies `preemption-recompute` and priorities under a single total order: the rank tuple `(priority, arrival_number)`. Store the waiting queue in a min-heap keyed by rank and keep sequences holding memory in a list sorted by rank. The decode memory pass walks the holding list from front to back and preempts victims from the back. Because victims always have the worst rank, a sequence already served earlier in the pass is never evicted, and the pass stops cleanly if the requesting sequence itself becomes the victim. When all priorities are equal, rank order reduces to arrival order, matching the prerequisite problem. The watermark leaves headroom for active sequences to cross block boundaries without immediately triggering preemption.

**Stage 5** scans the at most `max_running` sequences currently holding memory to find the donor with the longest shared full-block prefix, capped by what the donor has already prefilled and by `prompt_tokens - 1`. (Production engines with thousands of cached blocks use a hash chain as in `block-hash-prefix-cache` and keep unreferenced blocks in an LRU eviction queue.) Because shared prefix blocks are always full, copy on write never triggers inside `Engine`. Copy on write is still needed for full-sequence forks such as parallel sampling. Reference counting handles cleanup automatically when sequences finish or are preempted.

## Reference solution (Python)

```python
def _plan(self):
    sim, allocator, block = self.sim, self.allocator, self.block_size

    # 1. Decodes, best rank first; victims come from the back of the list.
    decode, evicted, i = [], False, 0
    while i < len(self._holding):
        request = self._holding[i][1]
        i += 1
        if sim.remaining_prefill(request.id):
            continue  # still prefilling: its blocks were taken at admission
        if sim.kv_tokens(request.id) % block == 0:
            while allocator.free_blocks() < 1 and len(self._holding) >= i:
                self._preempt(self._holding.pop())
                evicted = True
            if len(self._holding) < i:
                break  # the sequence that asked was itself the worst ranked
        allocator.append(request.id, 1)
        decode.append(request.id)
    budget = self.max_batch_tokens - len(decode)

    # 2. Prompts that are already being prefilled continue, in rank order.
    prefill = []
    for _, request in self._holding:
        if budget == 0:
            break
        remaining = sim.remaining_prefill(request.id)
        if remaining:
            budget -= self._chunk(prefill, request.id, remaining, budget)

    # 3. Admission from the head of the queue, unless this step had to preempt.
    while not evicted and budget > 0 and self._waiting and len(self._holding) < self.max_running:
        rank, request = self._waiting[0]
        if request.id in self._preempted:
            donor, shared = None, 0
            remaining = sim.remaining_prefill(request.id)  # prompt + produced tokens
        else:
            donor, shared = self._donor(request)
            remaining = request.prompt_tokens - shared
        need = -(-(shared + remaining) // block) - shared // block
        headroom = self.watermark_blocks if self._holding else 0
        if allocator.free_blocks() - need < headroom:
            break  # nobody overtakes the head of the queue
        heapq.heappop(self._waiting)
        if request.id in self._preempted:
            self._preempted.discard(request.id)
        else:
            if shared:
                allocator.fork(donor, request.id, shared)
            sim.admit(request, cached_tokens=shared)
        allocator.append(request.id, remaining)  # the whole prompt's blocks, now
        self._holding.append((rank, request))
        self._holding.sort(key=lambda entry: entry[0])
        budget -= self._chunk(prefill, request.id, remaining, budget)
    return prefill, decode
```

`run` waits when idle, calls `_plan`, awaits `sim.step`, and calls `allocator.free` for every finished sequence. The full reference also implements `BlockAllocator` and two helpers: `_preempt` (which frees blocks, calls `sim.preempt`, and pushes the request back onto `_waiting` with its original rank) and `_donor` (which computes $h(r, s)$ and selects the best donor).

The execution trace is deterministic because every tie is broken by a total order (rank for sequences, lowest id for blocks) and every state value comes directly from `sim_engine` or `allocator`.

## Common mistakes

- **Allocator leaks on failed `append`.** Popping blocks one by one before checking total capacity leaves blocks missing when `OutOfBlocks` is raised. Compute the required block count first and check it before popping anything.
- **Engine leaks on completion.** Forgetting to call `allocator.free(id)` for `result.finished` leaks blocks until the pool empties, which immediately fails the per-step allocator audit.
- **Missing copy on write or aliased block tables.** Sharing the parent's list object in `fork` causes one sequence's `append` to mutate the other's block table. Always copy the list and copy partially filled shared blocks on write.
- **One-step idle slot after completion.** Failing to remove finished sequences before checking `len(self._holding) < max_running` leaves a batch slot empty for one step after every completion.
- **Prefill starving decodes.** Giving the token budget to prompts before decodes blocks active sequences from generating tokens during long prefills.
- **Waiting for a whole prompt to fit.** A prompt longer than `max_batch_tokens` never fits in one step and must be chunked.
- **Preemption thrashing.** Ignoring `watermark_blocks` or admitting new requests during a step that just preempted hands freed blocks straight to new prompts. The pool fills back up and triggers another preemption at the next block boundary, multiplying recomputed tokens severalfold on the stage 4 workload.
- **Ordering victims or queues by arrival alone.** Preempting the newest sequence regardless of priority evicts urgent requests to keep lower-priority ones running.
- **Freeing shared blocks while still referenced.** Copying block ids in a prefix fork without incrementing reference counts returns blocks to the free pool when the donor finishes, letting another sequence overwrite live KV entries.
- **Sharing unprefilled or invalid prefixes.** Counting blocks a donor allocated but has not yet prefilled, rounding the match length up instead of down to full blocks, or sharing all `prompt_tokens` (leaving zero tokens to produce the first output token) violates the stage 5 contract.

## Complexity

Let $R \le \texttt{max\_running}$ be the number of sequences holding memory, $W$ be the waiting queue size, $N$ be `n_blocks`, and $L$ be the prompt length. Planning one step takes $O(R \log R)$ time plus $O(\log W)$ per admission, and stage 5 adds $O(R \cdot L)$ token comparisons per admission. Each allocator operation takes $O(b \log N)$ time for $b$ blocks touched. In simulation, the dominant metrics are recomputed tokens in stage 4 and prefilled tokens plus peak block usage in stage 5.

## In the wild

- Paged KV memory with reference-counted blocks and copy on write was introduced in "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arXiv:2309.06180), the paper behind vLLM.
- Iteration-level continuous batching was introduced in "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022).
- Prioritizing decodes and filling the remaining per-step token budget with prompt chunks is the stall-free batching design of "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (arXiv:2403.02310). vLLM exposes these controls as `max_num_batched_tokens` and `long_prefill_token_threshold`.
- vLLM preempts by recompute, freeing the victim's blocks and returning the request to the front of the waiting queue, and its priority policy selects the lowest-priority running sequence as the victim. Its automatic prefix caching uses the hash-chained, reference-counted block index from `block-hash-prefix-cache`.
