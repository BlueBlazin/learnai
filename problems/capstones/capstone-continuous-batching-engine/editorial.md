## The idea

Five components, one loop. The design question of the capstone is where the truth
about memory lives, and the answer that makes every stage short is: in the
allocator, updated **eagerly**. Whenever the plan decides that a sequence will hold
more tokens (an admission, a decode), the allocator call happens at that moment,
before the step runs. Then "how many blocks are available" is never a calculation
with promises and reservations; it is `allocator.free_blocks()`.

![The planner builds each step from the queue and the allocator's blocks, and the kit's `SimEngine` runs it.](assets/architecture.svg)

**Stage 1** is the allocator of `paged-kv-copy-on-write`: a min-heap of free ids, a
reference count per block, a table and a token count per sequence. An append
computes how many blocks it needs (one for a copy, if its partly filled last block
is shared, plus growth), checks that against the free count *before touching
anything*, and only then pops. `free` decrements and returns a block to the heap
when its count reaches zero. The prefix fork of stage 5 is the same loop over a
slice of the parent's table.

**Stage 2** is `continuous-batching-scheduler` with the allocator attached at three
points: `append(id, prompt_tokens)` at admission, `append(id, 1)` per decode,
`free(id)` per finished sequence. Taking the whole prompt's blocks at admission is
what later lets a prompt be prefilled in chunks without ever waiting for memory in
the middle.

**Stage 3** is the planner of `chunked-prefill-scheduling` run in three passes that
share one budget: decodes, admitted sequences with prefill left, admissions. A
chunk is `min(remaining, threshold, budget)`. Decodes first is the whole point:
the budget bounds the length of a step, and a sequence that is generating gets a
token in every step.

**Stage 4** is `preemption-recompute`, generalized by one idea: a single total
order, the rank `(priority, arrival number)`. The waiting queue is a heap by rank;
the sequences that hold memory are a list sorted by rank. The memory pass walks
that list from the front and takes victims from the back, so a sequence that has
been served in this step can never be the victim, and the pass ends when the
sequence that asked is itself the last one. With equal priorities the rank is the
arrival order, a victim is the newest sequence, and a preempted sequence is first
in the queue: exactly the prerequisite. The watermark is admission control against
the engine's own future: sequences grow by a block every `block_size` tokens, and
a pool admitted to the brim preempts at the next boundary.

**Stage 5** needs no index at all, because the only blocks worth sharing are held
by at most `max_running` sequences: compare the new prompt with each holder's,
cap the match by what the holder has really prefilled and by `prompt_tokens - 1`,
round down to blocks, and `fork` from the best. (A real engine with thousands of
cached blocks uses the hash chain of `block-hash-prefix-cache` for the same
lookup, and keeps unreferenced blocks in an eviction queue instead of freeing
them.) Shared blocks are always full, so copy on write never fires in the engine:
it exists for the whole forks of parallel sampling. Reference counts do the rest:
finishing or preempting a sequence frees only what nobody else holds.

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

`run` is the idle wait, `_plan`, the step, and `allocator.free` for every finished
id. The full file also has the allocator and the two helpers (`_preempt`: free,
`sim.preempt`, push back with the same rank; `_donor`: the formula for $h$).

Why the trace is unique: every choice is a minimum over a total order (rank for
sequences, id for blocks), and every quantity the plan reads is either the
engine's (`kv_tokens`, `remaining_prefill`) or the allocator's.

## Common mistakes

- **Leaked blocks, allocator.** Popping blocks one at a time and raising when the
  heap runs dry: the refused append has already taken blocks. Count first.
- **Leaked blocks, engine.** No `free` for `result.finished`. Nothing fails until
  the pool is empty, which is why the allocator is audited at every step.
- **No copy on write**, or a fork that hands the child the parent's list object:
  one sequence's append then changes the other's table.
- **An idle slot.** Counting a sequence that just finished against `max_running`
  while planning the next step leaves its slot empty for one step in every finish.
- **Prefill starving decodes.** Giving prompts the budget first: with a tight
  budget the running sequences get no token until the prompt is through.
- **Waiting until the prompt fits.** A prompt longer than the budget never fits;
  split it.
- **Preemption thrash.** Ignoring the watermark, or admitting in the step that
  preempted: the freed blocks go to a new prompt and the pool is full again when
  the next sequence crosses a block boundary. On the stage 4 workload the
  recomputed tokens rise severalfold.
- **Victim or queue by arrival only.** Preempting the newest sequence whatever
  its priority evicts an urgent request to keep a background one.
- **A shared block freed while referenced.** A prefix fork that copies table
  entries without incrementing their counts: when the donor finishes, its blocks
  return to the pool, the next admission takes them, and two sequences now hold
  the same block for different tokens.
- **Sharing what is not there.** Counting blocks the donor has allocated but not
  yet prefilled, rounding the match up to a block, or sharing the whole prompt and
  leaving no token whose forward pass produces the first output.

## Complexity

Planning a step is $O(R \log R)$ for $R \le$ `max_running` sequences in memory plus
$O(\log W)$ per admission from a queue of $W$; stage 5 adds $O(R \cdot L)$ token
comparisons per admission for prompts of length $L$. Allocator calls are
$O(\text{blocks touched} \cdot \log N)$. The costs that matter are the simulated
ones: recomputed tokens (stage 4) and prefilled tokens and peak blocks (stage 5).

## In the wild

- Paged KV memory with reference-counted blocks and copy on write is
  PagedAttention, from "Efficient Memory Management for Large Language Model
  Serving with PagedAttention" (arXiv:2309.06180), the paper behind vLLM.
- Iteration-level scheduling was introduced by "Orca: A Distributed Serving System
  for Transformer-Based Generative Models" (OSDI 2022).
- Decodes first, then prompt chunks under a per-step token budget, is the
  stall-free batching of "Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve" (arXiv:2403.02310); vLLM's scheduler exposes the budget as
  `max_num_batched_tokens` and the per-prompt cap as `long_prefill_token_threshold`.
- vLLM preempts by recompute: it frees the victim's blocks and puts the request
  back at the front of the waiting queue, and with its priority scheduling policy
  the victim is the lowest-priority running request. Its automatic prefix caching
  is the hash-chained, reference-counted block index of `block-hash-prefix-cache`.
