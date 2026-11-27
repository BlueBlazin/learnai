## The idea

**Why memory runs out after admission.** A sequence's KV cache grows by one token per decode step until generation ends, and the engine cannot predict that length in advance. Admission can verify that a prompt fits right now, but it cannot guarantee that eight running sequences will still fit thirty steps later. Any engine that admits requests optimistically therefore needs a recovery mechanism when a running sequence needs a block and none is free.

**Reclaiming memory while preserving output.** To free space, the engine evicts all KV blocks belonging to one running sequence. No tokens already shown to the client are lost because the generated output tokens stay attached to the request. Only derived state—the keys and values—is discarded. When space opens up, the engine rebuilds that KV cache by running the original prompt *plus all generated tokens* through the model as a single prefill. Because that prefill ends on the last generated token, its forward pass produces the next output token, and normal decoding resumes from there. The exact compute overhead is the number of tokens prefilled a second time.

**Choosing the victim.** The policy preempts the most recently admitted running sequence. On average, it has the shortest context and therefore the fewest tokens to recompute. Evicting the newest sequence also preserves first-come, first-served fairness so a newer arrival never sets back an older request. For the same reason, the victim is placed at the *front* of the waiting queue because it arrived earlier than every other waiting request.

**Step planning order.** Running sequences are processed first, from oldest to newest, because they have priority over available memory. Only leftover blocks are offered to waiting requests. Iterating through the running list from oldest to newest while popping victims from the back handles a subtle edge case cleanly. If the sequence currently requesting a block is itself the last element in the running list, it preempts itself and the loop over running sequences terminates.

**The watermark.** Without headroom, admission fills the KV pool to the very last block. On the next step, as soon as any sequence crosses a block boundary, the scheduler must preempt the sequence it just admitted. That prefill work is wasted, and the same cycle repeats when the victim returns to a full pool, causing thrashing. Requiring `watermark_blocks` free blocks after admission gives running sequences room to grow. The watermark must be waived when nothing else is scheduled in the step, or a large request that needs nearly the entire pool could never start.

**Why nothing is admitted after a preemption.** This rule is also a mathematical consequence of the block counts. Preemption only triggers when zero blocks are available. Evicting a victim frees its $b$ blocks, and the sequence that triggered preemption immediately takes 1 of them, leaving at most $b - 1$ blocks available. Because the victim is now at the front of the waiting queue and needs at least $b$ blocks to resume ($b + 1$ if it preempted itself at a block boundary), it cannot fit in the same step.

## Reference solution (Python)

```python
import asyncio
from collections import deque


async def run(engine, queue, max_running, watermark_blocks):
    block = engine.block_tokens
    waiting = deque()
    running = []
    preempted = set()

    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        reserved = 0
        evicted = False
        decode = []
        for request in running[:]:
            if request not in running:
                break
            if engine.kv_tokens(request.id) % block == 0:
                while engine.free_blocks - reserved < 1:
                    victim = running.pop()
                    engine.preempt(victim.id)
                    preempted.add(victim.id)
                    waiting.appendleft(victim)
                    evicted = True
                    if victim is request:
                        break
                if request not in running:
                    break
                reserved += 1
            decode.append(request.id)

        prefill = []
        while not evicted and waiting and len(running) < max_running:
            request = waiting[0]
            if request.id in preempted:
                tokens = engine.remaining_prefill(request.id)
            else:
                tokens = request.prompt_tokens
            need = -(-tokens // block)
            headroom = watermark_blocks if decode or prefill else 0
            if engine.free_blocks - reserved - need < headroom:
                break
            waiting.popleft()
            if request.id in preempted:
                preempted.discard(request.id)
            else:
                engine.admit(request)
            reserved += need
            prefill.append((request.id, tokens))
            running.append(request)

        result = await engine.step(prefill=prefill, decode=decode)
        finished = set(result.finished)
        running = [request for request in running if request.id not in finished]
```

The `reserved` counter tracks the one piece of state the engine cannot provide during planning. Because the engine allocates blocks only when `engine.step` begins, `engine.free_blocks` does not yet reflect blocks promised earlier in the same planning pass.

## Common mistakes

- **Preempting the oldest or largest sequence.** The oldest request has waited the longest and has the most context to recompute. Under sustained load, repeatedly evicting it causes starvation.
- **Restarting the victim as a new request.** Calling `engine.admit` again restarts generation from token 0, producing and streaming the initial tokens a second time.
- **Recomputing only the prompt.** A resumed sequence's context includes every output token it has already produced. Omitting them corrupts the context for the next token, and the simulated engine refuses to decode it.
- **Pushing the victim to the back of the waiting queue.** This lets newer arrivals overtake a request that was already running.
- **Omitting the watermark.** On the statement's 120-request workload, dropping the watermark raises recomputed tokens from 1759 to 5527 and multiplies preemptions by more than three and a half.
- **Applying the watermark everywhere.** Enforcing the watermark on running sequences evicts active requests while blocks are still free. Enforcing it on an empty step prevents a lone large request from ever starting.
- **Checking admission against `engine.free_blocks` alone.** Ignoring blocks already promised to decoding sequences in the same step triggers `OutOfBlocks`.
- **Off-by-one boundary check.** A new block is needed when `engine.kv_tokens(id)` is a multiple of `engine.block_tokens` *before* the step, because the step adds one token.
- **Letting a small waiting request overtake a large one.** Skipping the head of the queue keeps the pool full and starves the large request indefinitely.

## Complexity

Planning a step takes $O(\text{running} + \text{admitted})$ engine queries plus one operation per preemption, where each preemption removes a sequence from the running list. The dominant runtime cost happens inside the engine, where preempting a sequence of context length $c$ adds $c$ recomputed prefill tokens later.

## In the wild

- **vLLM** triggers preemption when KV cache space is insufficient for all batched requests and recomputes preempted requests once space returns. Its logs report `Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space`, and its tuning guide recommends adding KV memory or lowering `max_num_seqs` or `max_num_batched_tokens` (vLLM documentation, "Optimization and Tuning", section "Preemption"). Earlier versions also supported swapping KV blocks to CPU memory and back, but recompute is the default in the current engine.
- In vLLM's scheduler (`vllm/v1/core/sched/scheduler.py`), the loop over running requests asks the KV cache manager for blocks one request at a time. When allocation fails under FCFS, the scheduler evicts the last element of the running list. Under priority scheduling, it evicts the request with the largest `(priority, arrival_time)` tuple (lowest priority, then latest arrival). The victim's blocks are freed, its computed token count resets to zero, and it is prepended to the waiting queue while its output tokens are preserved and replayed. Any step that preempts skips admitting waiting requests, and the KV cache manager applies its `watermark` fraction only to waiting or preempted requests when other work is already scheduled in the step.
- Real serving engines add three features omitted here: request priorities (victims are chosen by priority before arrival time), chunked prefill (a resumed sequence can be recomputed across multiple steps), and prefix caching (blocks shared with other sequences are not freed on preemption, and a resumed request may hit cached prefix blocks).
