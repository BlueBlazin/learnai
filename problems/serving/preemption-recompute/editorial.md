## The idea

**Why memory runs out after admission.** A sequence's KV cache grows by one token
per step until an end the engine cannot foresee. Admission can check that a prompt
fits now; it cannot check that eight sequences will still fit thirty steps from
now. Every engine that admits optimistically therefore needs an answer to "a
running sequence needs a block and there is none".

**The answer: take memory from someone, keep their output.** The engine frees all
blocks of one running sequence. Nothing the client has seen is lost: the tokens
already produced stay with the request. What is lost is derived data, the keys and
values, and it can be rebuilt by running the prompt *and the produced tokens*
through the model again as one long prefill. That prefill ends on the last produced
token, so its output is the next token, and generation continues where it stopped.
The cost is exactly the tokens prefilled twice.

**Who.** The most recently admitted sequence. It has the least invested (the
shortest context, on average, and the fewest tokens to recompute), and choosing it
preserves first come, first served: an older request is never set back by a newer
one. For the same reason the victim goes to the *front* of the waiting queue. It
arrived before everything else that is waiting.

**The order of a step.** Running sequences first, oldest first, because they have
priority over the memory; only what is left can be offered to waiting requests.
Processing the running list in order while popping victims from its end has a neat
consequence: when the sequence being served is itself the last one, it preempts
itself, and the loop over the running list is finished.

**The watermark.** Without headroom, admission fills the pool to the last block,
and the very next step, in which some sequence crosses a block boundary, preempts
the sequence that was just admitted. Its prefill was wasted, and it will be wasted
again when it returns into an equally full pool: thrashing. Requiring a few free
blocks after admitting gives the running sequences room to grow. The watermark
must not apply to a step that would otherwise be empty, or a request that needs
nearly the whole pool could never start.

**Nothing is admitted in a step that preempts.** This is stated as a rule and is
also a theorem. A preemption happens only when no block is available, and frees
the $b$ blocks of the victim, of which the sequence that needed one takes 1. The
victim is now first in line and needs at least $b$ blocks to return ($b + 1$ if it
preempted itself, since it was at a block boundary). At most $b - 1$ are
available.

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

`reserved` is the one piece of bookkeeping the engine does not do for you: blocks
are taken when the step starts, so while a step is being planned, the engine's
`free_blocks` does not yet know about the blocks this step has promised.

## Common mistakes

- **Preempting the oldest**, or the largest. The oldest has the most work to redo
  and is the request that has waited longest; under sustained pressure it is
  preempted again and again and never finishes.
- **Restarting the victim** as a new request. Its first tokens are generated, and
  streamed, a second time.
- **Recomputing the prompt only.** The context of a resumed sequence includes what
  it has produced; without those tokens the next token is predicted from the wrong
  context (here the engine refuses to decode it).
- **Sending the victim to the back of the queue**, which lets later arrivals
  overtake a request that was already running.
- **No watermark**: 5527 recomputed tokens instead of 1759 on the statement's
  workload, three and a half times the preemptions.
- **Applying the watermark everywhere**: to running sequences (preempting to keep
  blocks free, which evicts sequences while there is room) or to a lone request
  (which then can never be admitted).
- **Planning admission against `engine.free_blocks` alone**, forgetting the blocks
  promised to decoding sequences in the same step: `OutOfBlocks`.
- **The boundary test off by one**: the block is needed when `kv_tokens` is a
  multiple of the block size *before* the step, since the step adds one token.
- **Letting a small request overtake** a waiting large one. It keeps the pool full
  and the large request waits forever.

## Complexity

Planning a step is $O(\text{running} + \text{admitted})$ engine queries plus the
preemptions, each of which removes one running sequence. The cost that matters is
in the engine: every preemption of a sequence with context $c$ adds $c$ prefill
tokens later.

## In the wild

- **vLLM** preempts when "KV cache space is insufficient to handle all batched
  requests", and "preempted requests are recomputed when sufficient KV cache space
  becomes available again"; the log line is "Sequence group 0 is preempted by
  PreemptionMode.RECOMPUTE mode because there is not enough KV cache space", and
  the documentation's remedies are more KV memory or a lower `max_num_seqs` or
  `max_num_batched_tokens` (vLLM documentation, "Optimization and Tuning",
  section "Preemption"). The alternative, swapping the blocks to CPU memory and
  back, existed in earlier versions; recompute is the default in the current
  engine.
- In vLLM's scheduler (`vllm/v1/core/sched/scheduler.py`) the loop over running
  requests calls the KV cache manager for each; when allocation fails, the victim
  under first come, first served is the last element of the running list, and
  under the priority policy the request with the largest `(priority,
  arrival_time)`, that is, the least important and among those the latest. The
  victim's blocks are freed, its count of computed tokens is reset to zero, and it
  is prepended to the waiting queue; its output tokens are kept and replayed. A
  scheduling step in which a preemption occurred does not admit waiting requests.
  The KV cache manager has a `watermark` (a fraction of the pool) that is applied
  only to waiting or preempted requests and only when other requests are scheduled
  in the step, which is the rule used here.
- This problem leaves out priorities (the victim would be chosen by priority
  first), chunked prefill (a resumed sequence could be recomputed over several
  steps), and prefix caching (blocks shared with other sequences are not freed by
  a preemption, and a resumed sequence may find its prefix still cached).
