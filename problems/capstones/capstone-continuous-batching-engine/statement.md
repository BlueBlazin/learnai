Build the scheduling core of an LLM inference engine in five stages. You will implement a paged KV cache with reference-counted blocks, a continuous-batching loop on top of it, chunked prefill under a per-step token budget, priority scheduling with preemption by recompute when memory runs out, and prompt-prefix block sharing between requests.

## Overview

```python
class OutOfBlocks(Exception): ...


class BlockAllocator:  # stage 1; fork(..., n_tokens) in stage 5
    def __init__(self, n_blocks, block_size): ...
    def append(self, seq_id, n_tokens): ...  # -> (allocated, copies)
    def fork(self, parent_id, child_id, n_tokens=None): ...  # -> None
    def free(self, seq_id): ...  # -> None
    def block_table(self, seq_id): ...  # -> list of block ids
    def ref_count(self, block): ...  # -> int
    def free_blocks(self): ...  # -> int


class Engine:  # stages 2 to 5
    def __init__(
        self,
        sim_engine,
        n_blocks,
        block_size,
        max_batch_tokens,
        *,
        max_running=8,
        long_prefill_threshold=None,
        watermark_blocks=0,
        prefix_sharing=False,
    ): ...

    allocator: BlockAllocator

    def submit(self, request) -> None: ...
    async def run(self) -> None: ...
```

![The planner builds each step from the queue and the allocator's blocks, and the kit's `SimEngine` runs it.](assets/architecture.svg)

Test cases are grouped by stage. Each stage uses only the features defined in that stage and earlier ones, because later features default to off (`long_prefill_threshold=None`, `watermark_blocks=0`, `prefix_sharing=False`, and `priority=0`) or are non-binding. You can implement and pass the stages in order.

### The simulated engine

`sim_engine` is the kit's [`SimEngine`](../../../python/src/learnai/kit/simengine.py), which simulates forward passes in virtual time measured in **milliseconds**. `Engine` interacts with it through four methods:

- `sim_engine.admit(request, cached_tokens=0)` registers a request as a live sequence. It takes zero virtual time.
- `result = await sim_engine.step(prefill=[(id, n), ...], decode=[id, ...])` runs one forward pass over $P$ total prompt tokens and $D$ decoded sequences. One step takes $8 + 0.25\,P + D$ ms. A prompt can be prefilled in one step or in chunks across several steps. The step that finishes a sequence's prefill produces its **first** output token. On later steps, that sequence appears in `decode` and produces one token per step. A sequence id may appear at most once across `prefill` and `decode` in a single step. `result.finished` contains the sequence ids that produced their final token and are no longer live.
- `sim_engine.remaining_prefill(id)` returns how many tokens the sequence still needs to prefill before it can decode (`0` means ready). `sim_engine.kv_tokens(id)` returns how many tokens the sequence currently holds in the KV cache. A prefill of `n` adds `n` tokens and a decode adds `1` token when the step **starts**, so a ready sequence that has produced $g$ tokens holds $\texttt{prompt\_tokens} + g - 1$ tokens in the KV cache.
- `sim_engine.preempt(id)` (stage 4) discards the sequence's KV cache while keeping the sequence live and preserving the output tokens it already produced. Its full context (prompt plus produced tokens) must be prefilled again, and `remaining_prefill(id)` reports that total. The step that finishes this re-prefill produces its *next* output token. Do not call `admit` on a preempted sequence again.

`SimEngine` enforces no memory limit of its own and runs one step at a time. Any invalid call raises `ProtocolViolation` and fails the test case even if caught. Invalid calls include an empty step, a duplicate id in a step, decoding a sequence with incomplete prefill, prefilling more than `remaining_prefill(id)`, exceeding `max_batch_tokens` total tokens in a step, or referencing an unknown id.

### Requests

Requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Read `id` (hashable and unique), `prompt_tokens`, `priority` (stage 4), and `tokens` (stage 5). Do not read `output_tokens`, because a real engine only discovers an output's length when the sequence finishes.

## Stage 1: the block allocator

The memory pool contains `n_blocks` physical blocks numbered `0` to `n_blocks - 1`, and each block holds `block_size` tokens. A **live sequence** has a token count $t \ge 1$ and a **block table** listing the physical block ids that store its tokens in logical order. Token $i$ lives at slot `i % block_size` of block `table[i // block_size]`. Every block table therefore has length $\lceil t / \texttt{block\_size} \rceil$, and only its last block can have empty slots. Multiple sequences may reference the same physical block. A block's **reference count** is the number of live sequences whose block table contains it, and a block is free if and only if its reference count is 0. The allocator stores no tensors.

- `append(seq_id, n_tokens)` appends `n_tokens` $\ge 1$ tokens to `seq_id`. If `seq_id` is not live, it starts with $t = 0$ tokens. Let $t$ be the token count before the call, and let $n = \texttt{n\_tokens}$. The call performs two steps:
  1. **Copy on write.** If the last block has empty slots ($t \bmod \texttt{block\_size} \ne 0$) and its reference count is greater than 1, the sequence cannot write into that shared block. Allocate a new block `dst`, replace the last entry `src` of this sequence's table with `dst`, and decrement `src`'s reference count by 1. A full last block is never written to, so it is never copied.
  2. **Growth.** Allocate $\lceil (t + n) / \texttt{block\_size} \rceil - \lceil t / \texttt{block\_size} \rceil$ additional blocks and append them to the table.

  Allocate free blocks **lowest id first**, taking the copy destination (if any) before the growth blocks. Return `(allocated, copies)`, where `allocated` is the list of newly allocated block ids in allocation order and `copies` is `[(src, dst)]` if a copy occurred or `[]` otherwise. If fewer free blocks are available than needed, raise `OutOfBlocks` and **change nothing**: no blocks are allocated, and a non-live `seq_id` remains non-live.
- `fork(parent_id, child_id)` creates `child_id` with the same token count as `parent_id` and a new list containing the same block ids as `parent_id`'s table. Every block in that table gains 1 reference, and no new blocks are allocated. Raise `ValueError` if `child_id` is already live (including `child_id == parent_id`). Otherwise raise `KeyError` if `parent_id` is not live. Leave all state unchanged on error. (The `n_tokens` parameter is added in stage 5.)
- `free(seq_id)` removes `seq_id`, decrements the reference count of every block in its table by 1, and returns any block whose count reaches 0 to the free pool. Raise `KeyError` if `seq_id` is not live.
- `block_table(seq_id)` returns a fresh copy of `seq_id`'s block table, raising `KeyError` if `seq_id` is not live. `ref_count(block)` returns the reference count of a block id in `[0, n_blocks)`. `free_blocks()` returns the number of blocks with reference count 0.

### Example 1

```python
alloc = BlockAllocator(n_blocks=6, block_size=4)
alloc.append("a", 6)  # ([0, 1], [])      block 1 holds 2 of 4 tokens
alloc.fork("a", "b")  #                   b: table [0, 1]; both counts are 2
alloc.append("b", 1)  # ([2], [(1, 2)])   block 1 is shared and partly filled:
#                                         b moves to a copy, b: [0, 2]
alloc.append("a", 1)  # ([], [])          a is block 1's only holder now: in place
alloc.append("b", 4)  # ([3], [])         b: 11 tokens, table [0, 2, 3]
alloc.fork("b", "c")  #                   ref_count(0) is 3
alloc.free("b")  #                        c holds all of b's blocks: free_blocks() is 2
alloc.append("c", 2)  # ([4], [])         block 3 (3 of 4) is c's alone, then growth
alloc.free("a")  #                        block 1 returns; block 0 is still c's
alloc.free("c")  #                        free_blocks() is 6
```

## Stage 2: continuous batching

`Engine` creates `self.allocator = BlockAllocator(n_blocks, block_size)` and uses it as the engine's KV memory.

The test harness calls `submit(request)` synchronously from an event-loop callback at each request's arrival time. `submit` appends the request to the waiting queue, wakes `run`, and returns. `run()` is an async coroutine that never returns. A request is *waiting* until admitted, and an admitted sequence *holds memory* until it finishes (or is preempted in stage 4). Before every step, `run` plans work in three stages:

1. **Decodes.** Every sequence that holds memory and has completed prefill decodes in this step. Call `allocator.append(id, 1)` for each decoding sequence to reserve room for the new token.
2. **Admits.** Admit waiting requests first come, first served while fewer than `max_running` sequences hold memory and at least one request is waiting. To admit a request, call `sim_engine.admit(request)` and `allocator.append(id, prompt_tokens)`. Each sequence takes the blocks for its **entire prompt at admission** and prefills its prompt in this step.
3. **Step.** Call `await sim_engine.step(prefill=..., decode=...)`. `prefill` lists the sequences admitted in step (2) in admission order, each paired with its full `prompt_tokens`. `decode` lists the sequences from step (1) in admission order.

When `step` returns, call `allocator.free(id)` for every `id` in `result.finished` so those sequences stop holding memory. Plan the next step immediately at the same virtual timestamp. When no request is waiting and no sequence holds memory, `run` blocks (for example on an `asyncio.Event`) until `submit` wakes it. Never issue an empty step, sleep, or poll.

### Events at the same instant

Requests that arrive at the same virtual timestamp are submitted in arrival order before `run` resumes. Any request that arrives during a step, or at the exact instant a step finishes, is already in the waiting queue when the next step is planned.

### Per-step allocator audit

Whenever `sim_engine.step` is called, the test harness checks three invariants:

- For every sequence holding memory, `len(allocator.block_table(id))` equals $\lceil T / \texttt{block\_size} \rceil$, where $T$ is `kv_tokens(id) + remaining_prefill(id)` before the step, plus `1` if the sequence decodes in this step.
- Every block's reference count equals the number of live sequences holding memory whose table contains that block.
- `free_blocks()` plus the number of distinct blocks across those tables equals `n_blocks`, and at most `max_running` sequences hold memory.

Leaving a finished sequence unfreed leaks blocks and fails this audit. In stages 2 and 3, the pool is large enough that `append` never runs out of blocks.

### Example 2

With `max_running = 2` and requests given as `(arrival ms, prompt tokens, output tokens)`: request 0 `(0, 8, 4)`, request 1 `(0, 16, 2)`, request 2 `(5, 8, 3)`, request 3 `(45, 4, 2)`, request 4 `(50, 12, 1)`, and request 5 `(60, 4, 2)`.

| starts | prefill | decode | takes | at its end |
|---|---|---|---|---|
| 0 | `(0, 8), (1, 16)` | | 8 + 6 = 14 | first tokens of 0 and 1 |
| 14 | | `0, 1` | 8 + 2 = 10 | request 1 finishes |
| 24 | `(2, 8)` | `0` | 11 | first token of 2 |
| 35 | | `0, 2` | 10 | request 0 finishes |
| 45 | `(3, 4)` | `2` | 10 | request 2 finishes |
| 55 | `(4, 12)` | `3` | 12 | requests 4 and 3 finish |
| 67 | `(5, 4)` | | 9 | first token of 5 |
| 76 | | `5` | 9 | request 5 finishes |

Request 2 arrives at 5 ms and joins at 24 ms as soon as request 1 frees a slot, running alongside the decode of request 0. Request 3 arrives at 45 ms, the exact instant the previous step ends, so it enters the next step immediately.

## Stage 3: chunked prefill

Each step can process at most `max_batch_tokens` total tokens, counting both prompt tokens and decoded sequences. Planning a step now follows three passes that share a single token budget:

1. **Decodes first.** Schedule every ready sequence as in stage 2. Each decode uses 1 token of the budget. Because `max_running <= max_batch_tokens`, all ready sequences always fit.
2. **Prompts in progress.** Walk the sequences that hold memory and still have prefill remaining, in admission order, while budget remains. Assign each one a chunk of

   $$
   n = \min(\text{remaining prefill},\ \texttt{long\_prefill\_threshold},\ \text{budget left})
   $$

   tokens, omitting `long_prefill_threshold` when it is `None`.
3. **Admission.** While budget remains, fewer than `max_running` sequences hold memory, and a request is waiting, admit the oldest waiting request as in stage 2 (allocating all blocks for its full prompt now) and assign it a chunk using the same formula.

`prefill` lists the chunks from pass (2) followed by the chunks from pass (3), each in order. Prompts that do not fit in the remaining budget are split across steps, and no sequence is skipped. A sequence whose final prefill chunk runs in this step begins decoding on the next step. When the budget never binds and `long_prefill_threshold` is `None`, this matches stage 2.

### Example 3

With `max_batch_tokens = 32`, `long_prefill_threshold = 24`, and requests: request 0 `(0, 16, 4)`, request 1 `(0, 40, 2)`, and request 2 `(3, 8, 2)`.

| starts | prefill | decode | takes |
|---|---|---|---|
| 0 | `(0, 16), (1, 16)` | | 8 + 8 = 16 |
| 16 | `(1, 24), (2, 7)` | `0` | 8 + 7.75 + 1 = 16.75 |
| 32.75 | `(2, 1)` | `0, 1` | 10.25 |
| 43 | | `0, 2` | 10 |

At 0 ms, request 1 receives the 16 tokens left after request 0, while still reserving all 40 tokens' worth of blocks at admission. At 16 ms, decoding request 0 costs 1 token, request 1 is capped at 24 by `long_prefill_threshold`, and request 2 takes the remaining 7 tokens.

## Stage 4: priorities and preemption

The block pool can now run out of memory, and every request has an integer `priority` where **smaller values are more urgent**.

Define a request's **rank** as the tuple `(priority, arrival_number)`, where `arrival_number` counts `submit` calls starting from `0`. Smaller ranks are better. One total order governs the entire engine: the waiting queue, the sequences holding memory, and both `prefill` and `decode` lists are ordered by rank. A preempted sequence returns to the waiting queue with its original rank. When all priorities are equal, rank order reduces to arrival order, matching stages 2 and 3.

Planning a step now follows three passes:

1. **Memory for decodes, best rank first.** Iterate through sequences holding memory in rank order, skipping any whose prefill is incomplete (they already hold their prompt blocks). A ready sequence needs one new block if and only if `kv_tokens(id) % block_size == 0`. If it needs a block and `allocator.free_blocks() == 0`, **preempt the worst-ranked sequence currently holding memory**: call `allocator.free(victim)` and `sim_engine.preempt(victim)`, and return the victim to the waiting queue. Repeat until at least one block is free or the requesting sequence itself becomes the victim (because it had the worst rank). If it still holds memory, call `allocator.append(id, 1)` and add it to `decode`.
2. **Prompts in progress.** Assign chunks to sequences holding memory with remaining prefill, in rank order, as in stage 3.
3. **Admission (only if pass 1 preempted nothing).** While budget remains, fewer than `max_running` sequences hold memory, and the waiting queue is non-empty, inspect the head of the waiting queue (best rank). Let `prefill` be its `prompt_tokens`, or `remaining_prefill(id)` if it was previously preempted. It requires `needed` = $\lceil \text{prefill} / \texttt{block\_size} \rceil$ blocks. Admit it if

   `allocator.free_blocks() - needed >= watermark_blocks`

   or, when no sequence currently holds memory, if `allocator.free_blocks() - needed >= 0`. To admit it, call `allocator.append(id, prefill)`, call `sim_engine.admit(request)` if it has never been admitted before, and assign its first chunk. If the head cannot be admitted, stop admission immediately so lower-ranked requests cannot overtake it.

A preempted sequence keeps the tokens it already generated, and the tokens it must re-prefill are tracked in `sim_engine.counters.recomputed_tokens`. A victim may be a sequence that is still prefilling, and multiple sequences may be preempted during a single step.

### Example 4

With 7 blocks of 4 tokens, `max_running = 3`, `watermark_blocks = 0`, and requests `(arrival, prompt, output, priority)`: request 0 `(0, 8, 6, 1)`, request 1 `(0, 6, 7, 0)`, request 2 `(2, 7, 5, 2)`, and request 3 `(30, 4, 3, 0)`.

| starts | prefill | decode | free blocks during the step | notes |
|---|---|---|---|---|
| 0 | `(1, 6), (0, 8)` | | 3 | request 1 is more urgent, so it appears first in both lists |
| 11.5 | `(2, 7)` | `1, 0` | 0 | request 0 (8 tokens) takes a third block, and request 2 takes the last two |
| 23.25 | | `1, 0, 2` | 0 | |
| 34.25 | | `1, 0` | 1 | request 1 (8 tokens) needs a block and none is free. Request 2, the worst rank, is preempted after 2 tokens. No admission in this step |
| 44.25 | `(3, 4)` | `1, 0` | 0 | request 3 (priority 0) is ahead of request 2 in the queue and takes the free block |
| 55.25 | | `1, 3` | 2 | request 3 (4 tokens) needs a block. Request 3 is newest, but request 0 (priority 1) has the worst rank and is preempted after 5 of its 6 tokens |
| 65.25 | | `1, 3` | 2 | request 0 would need 4 blocks for its 13 tokens |
| 75.25 | `(0, 13), (2, 9)` | | 0 | requests 1 and 3 have finished. Request 0 produces its last token at 88.75 |
| 88.75 | | `2` | 4 | |
| 97.75 | | `2` | 4 | request 2 finishes at 106.75 |

In total, 22 tokens are recomputed. Request 2 generated tokens 0 and 1 before 34.25 ms and resumed with token 2 at 88.75 ms instead of starting over.

## Stage 5: prefix sharing

When `prefix_sharing=True`, a newly admitted request reuses KV blocks that a sequence currently holding memory has already prefilled for the same prompt prefix. `request.tokens` holds the prompt as a tuple of token ids, or `None` if the request does not participate in sharing.

### Partial fork

Extend `fork(parent_id, child_id, n_tokens=None)` so that `child_id` receives the parent's first `n_tokens` tokens. The child's token count becomes `n_tokens`, its block table is the first $\lceil \texttt{n\_tokens} / \texttt{block\_size} \rceil$ blocks of the parent's table, and each of those blocks gains 1 reference. Passing `n_tokens=None` forks all of the parent's tokens as in stage 1. After checking the stage 1 error conditions, raise `ValueError` if `n_tokens` is outside `[1, parent's token count]`. Copy-on-write behavior is unchanged: if a child partially occupies its last shared block, its first `append` copies that block.

### Donor selection at admission

When admitting a never-admitted request `r` whose `tokens` is not `None`, examine every sequence `s` that currently holds memory and has `tokens` not `None`. Compute the number of full shareable blocks $h(r, s)$:

$$
h(r, s) = \left\lfloor \frac{\min\bigl(\text{common}(r, s),\ \text{filled}(s),\ \texttt{r.prompt\_tokens} - 1\bigr)}{\texttt{block\_size}} \right\rfloor
$$

Here $\text{common}(r, s)$ is the length of the longest common prefix of `r.tokens` and `s.tokens`, and $\text{filled}(s) = \min(\texttt{kv\_tokens}(s),\ \texttt{s.prompt\_tokens})$ is how many prompt tokens `s` has already prefilled into the KV cache. Only **full blocks** that `s` has **already prefilled** can be shared, and at least one token of `r` must remain unshared so its forward pass produces the first output token.

Pick the **donor** `s` with the largest $h(r, s)$, breaking ties by best rank. If $h > 0$:

- Call `allocator.fork(donor, r.id, h * block_size)`.
- Call `sim_engine.admit(r, cached_tokens=h * block_size)` so `sim_engine` marks those tokens as cached and leaves `prompt_tokens - h * block_size` tokens to prefill.
- Call `allocator.append(r.id, prompt_tokens - h * block_size)` for the remaining prompt tokens. In the stage 4 admission check, the number of new blocks needed is therefore $\lceil \texttt{prompt\_tokens} / \texttt{block\_size} \rceil - h$.

Notice four consequences of these rules:

- Two requests admitted in the same step cannot share with each other because neither has prefilled yet.
- A preempted sequence returns without sharing (its full context is recomputed), though it can act as a donor again once it holds memory and has prefilled blocks.
- A donor may itself be sharing blocks from an earlier sequence, allowing cached prefix blocks to outlive the request that first computed them.
- Freeing or preempting a sequence returns only the blocks that no other sequence still references.

### Example 5

With 10 blocks of 4 tokens, `max_running = 4`, and `prefix_sharing=True`, let `S` be an 8-token system prompt `(1, 2, ..., 8)`. Five requests arrive:

- Request 0 at 0 ms: `S + (11, 12, 13)`, 4 output tokens.
- Request 1 at 0 ms: `S + (21, 22)`, 3 output tokens.
- Request 2 at 5 ms: `S + (31, ..., 35)`, 2 output tokens.
- Request 3 at 30 ms: `(1, 2, 3, 4, 9, 9, 9, 9, 9)`, 2 output tokens.
- Request 4 at 40 ms: `S + (41,)`, 2 output tokens.

| starts | prefill | decode | notes |
|---|---|---|---|
| 0 | `(0, 11), (1, 10)` | | admitted together, so nothing can be shared yet |
| 13.25 | `(2, 5)` | `0, 1` | requests 0 and 1 both hold `S` in two full blocks. Request 0 is the donor (best rank), so request 2 shares 8 tokens and takes 2 new blocks |
| 24.5 | | `0, 1, 2` | |
| 35.5 | `(3, 5)` | `0` | requests 1 and 2 finished. Request 3 shares 4 tokens (one block) with request 0 |
| 45.75 | `(4, 5)` | `3` | request 0 finished, but its first block lives on in request 3, which becomes the donor for request 4 (4 tokens in common) |
| 56 | | `4` | request 4 finishes at 65 |

Sharing prefills 36 tokens instead of 52 and uses at most 9 blocks. Without sharing, the same workload runs out of memory at 26.5 ms and preempts request 2.

## Constraints

- `1 <= block_size <= 64`, `1 <= n_blocks <= 4096`, `1 <= max_running <= 16`, `max_running <= max_batch_tokens`, and `0 <= watermark_blocks < n_blocks`. Every request satisfies `prompt_tokens + output_tokens - 1 <= n_blocks * block_size`, so any request running alone finishes. Prompts contain 1 to 4,000 tokens. Arrival times are multiples of 0.25 ms, so all timestamps are exact floats.
- **Stage 1.** Return values, block tables, reference counts, and `free_blocks()` are checked exactly after every call on scripted and random workloads, alongside independent invariant checks.
- **Stages 2 to 5.** The scheduling rules determine a unique trace of steps, admissions, and preemptions. Tests compare your trace exactly against the examples above, additional scripted schedules, and seeded bursty and Poisson workloads of up to 120 requests. Tests also verify that every request produces each token once in order, that no sequence is dropped, that the stage 2 allocator audit holds at every step, and that all blocks are free at the end.
- **Stage 3 stall bound.** Six sequences are generating when a 4,000-token prompt arrives, followed by three 24-token prompts, with `max_batch_tokens = 512`. The longest gap between consecutive tokens of any of the six sequences must be at most **150 ms** (the largest step takes $8 + 0.25 \cdot 506 + 6 = 140.5$ ms), both with and without `long_prefill_threshold = 128`. With `long_prefill_threshold = 128`, each short prompt must produce its first token within **120 ms** of arriving.
- **Stage 4 cost.** On 120 Poisson requests across three priorities (prompts of 16 to 160 tokens, outputs of 8 to 96 tokens) with 40 blocks of 16 tokens, `max_running = 8`, `max_batch_tokens = 128`, and `watermark_blocks = 4`, the engine must incur at most **3000** recomputed tokens and **30** preemptions. The reference recomputes 1365 tokens across 11 preemptions, whereas omitting the watermark causes several times as many.
- **Stage 5 cost.** On 120 requests sharing three prefixes of 64 to 160 tokens with 96 blocks of 16 tokens and `max_running = 8`, the engine must prefill at most **6000** tokens (`sim_engine.counters.prefill_tokens`; the reference prefills 4767 compared to 18655 without sharing), use at most **64** blocks at any step (reference 54 vs 96 without sharing), and trigger zero preemptions.
- Use only `asyncio`, `BlockAllocator`, and `sim_engine`. Time is virtual, so wall-clock timers and real sleeping do not apply.

## Why this matters

This scheduling loop sits at the core of vLLM, TGI, SGLang, and `llama.cpp`'s server. Continuous batching keeps GPU utilization high, and a per-step token budget bounds inter-token latency. Paged blocks allow optimistic admission without reserving maximum output lengths up front, preemption recovers when memory runs out, and reference counting lets thousands of requests share a single system prompt in memory.

Most real serving bugs occur at the seams between these mechanisms. A leaked block on a rare preemption path eventually exhausts memory, and an off-by-one slot count leaves a batch slot idle after every completion. An unchunked prompt stalls active streams, admitting down to zero free blocks triggers preemption thrashing, and freeing a shared block while another sequence still attends to it silently corrupts generated text.
