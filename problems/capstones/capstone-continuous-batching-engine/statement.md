Build the scheduling core of an LLM inference engine in five stages: a paged KV cache
with reference-counted blocks, a continuous-batching loop on top of it, chunked
prefill under a token budget, priorities with preemption by recompute when memory
runs out, and sharing of prompt-prefix blocks between requests. Each stage is a few
lines on its own; the capstone is making them agree with each other, so that the
loop never plans a step the allocator cannot back and the allocator never frees a
block the loop still counts on.

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

The cases are grouped by stage, and each stage's cases use only what that stage and
the ones before it specify: later features are off by default (`long_prefill_threshold=None`,
`watermark_blocks=0`, `prefix_sharing=False`, every `priority` 0) or cannot matter (a
budget and a pool that are large enough). So the stages can be solved, and pass, in
order.

**The simulated engine.** `sim_engine` is the kit's
[`SimEngine`](../../../python/src/learnai/kit/simengine.py): a forward pass with the
model taken out. Time is virtual and counted in **milliseconds**. What `Engine` uses:

- `sim_engine.admit(request, cached_tokens=0)` makes a request a live sequence. It
  takes no time.
- `result = await sim_engine.step(prefill=[(id, n), ...], decode=[id, ...])` runs one
  forward pass over $P$ prompt tokens in total and $D$ decoded sequences, which takes
  $8 + 0.25\,P + D$ ms. A prompt may be prefilled in one step or in chunks over
  several. The step that completes a sequence's prefill produces its **first** output
  token; from the next step on the sequence is listed in `decode` and produces one
  token per step. A sequence may appear only once in the two lists of a step.
  `result.finished` holds the ids that produced their last token: they are no longer
  live.
- `sim_engine.remaining_prefill(id)`: tokens still to prefill before the sequence can
  be decoded (0: ready). `sim_engine.kv_tokens(id)`: tokens the sequence has in the KV
  cache; a prefill of `n` adds `n` and a decode adds 1 when the step **starts**, so a
  ready sequence that has produced `g` tokens holds `prompt_tokens + g - 1`.
- `sim_engine.preempt(id)` (stage 4): the sequence loses its KV cache but stays live
  and keeps the tokens it has produced. Its whole context, prompt plus produced
  tokens, must be prefilled again (`remaining_prefill(id)` says how many tokens that
  is); the step that completes this produces its *next* token. Do not `admit` it
  again.

The kit engine has no memory limit of its own and one step runs at a time. Anything
it cannot do (an empty step, an id listed twice, a decode of a sequence whose prefill
is incomplete, a chunk larger than the remaining prefill, more than
`max_batch_tokens` tokens in a step, an unknown id) raises `ProtocolViolation` and
fails the case, also if the exception is caught.

**Requests** are [`SimRequest`](../../../python/src/learnai/kit/workloads.py)
objects. Read `id` (hashable, unique), `prompt_tokens`, `priority` (stage 4), and
`tokens` (stage 5). Do not read `output_tokens`: an engine learns the length of an
output when the sequence finishes.

## Stage 1: the block allocator

The pool has `n_blocks` physical blocks with ids `0` to `n_blocks - 1`, each with
room for `block_size` tokens. A **live sequence** has a token count $t \ge 1$ and a
**block table**: the ids of the blocks that hold its tokens, in logical order. Token
$i$ lives in slot `i % block_size` of block `table[i // block_size]`, so a table has
exactly $\lceil t / \texttt{block\_size} \rceil$ entries and only the last block can
have empty slots. One block may be in the tables of several sequences; its
**reference count** is the number of live sequences whose table contains it, and a
block is free exactly when the count is 0. The allocator stores no tensors.

- `append(seq_id, n_tokens)` adds `n_tokens` $\ge 1$ tokens. A `seq_id` that is not
  live starts a new sequence with zero tokens. With $t$ tokens before the call:
  1. **Copy on write.** If the last block has empty slots ($t \bmod
     \texttt{block\_size} \ne 0$) and its reference count is greater than 1, the
     sequence may not write into it: a new block `dst` is allocated and replaces
     the last entry `src` of this sequence's table, and `src` loses one reference.
     A full last block is never written, hence never copied.
  2. **Growth.** $\lceil (t + n) / \texttt{block\_size} \rceil - \lceil t /
     \texttt{block\_size} \rceil$ further blocks are allocated and appended to the
     table.

  Blocks are taken from the free blocks **lowest id first**: the copy destination
  (if any) first, then the growth blocks. Returns `(allocated, copies)`:
  `allocated` is the list of block ids taken by this call in allocation order, and
  `copies` is `[(src, dst)]` if the call copied, otherwise `[]`. If the call needs
  more blocks than are free, it raises `OutOfBlocks` and **changes nothing**: no
  block is taken, and a `seq_id` that was not live is still not live.
- `fork(parent_id, child_id)` creates `child_id` with the token count and the block
  table of `parent_id` (the same ids, in a list of its own). Every block of the
  table gains one reference; nothing is allocated. Raises `ValueError` if `child_id`
  is live (this includes `child_id == parent_id`), and otherwise `KeyError` if
  `parent_id` is not live; nothing changes. (`n_tokens` belongs to stage 5.)
- `free(seq_id)` forgets the sequence: each block of its table loses one reference,
  and a block whose count reaches 0 returns to the pool. Raises `KeyError` if
  `seq_id` is not live.
- `block_table(seq_id)` returns the table as a new list; `KeyError` if not live.
  `ref_count(block)` returns the reference count of a block id in `[0, n_blocks)`.
  `free_blocks()` returns the number of blocks with count 0.

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

`Engine` owns `self.allocator = BlockAllocator(n_blocks, block_size)`, and that
allocator is the engine's memory from here on.

`submit(request)` is called by the test at the request's arrival instant, from a
plain callback inside the event loop. It is a plain method: it puts the request into
the waiting queue, wakes the loop, and returns. `run()` is a coroutine that never
returns. A request is *waiting* until it is admitted; an admitted sequence *holds
memory* until it finishes (or, in stage 4, is preempted). Before every step, `run`:

1. **Decodes.** Every sequence that holds memory and whose prefill is complete
   decodes in this step. The token a decode stores needs room: call
   `allocator.append(id, 1)` for each.
2. **Admits**, first come, first served: while fewer than `max_running` sequences
   hold memory and a request is waiting, admit the oldest one. Admitting is
   `sim_engine.admit(request)` and `allocator.append(id, prompt_tokens)`: a sequence
   takes the blocks of its **whole prompt when it is admitted**. Its prompt is
   prefilled in this step.
3. **Steps** once: `prefill` lists the sequences admitted in (2), in admission
   order, each with its whole prompt; `decode` lists the sequences of (1), in
   admission order.

When the step returns, each sequence in `result.finished` stops holding memory:
`allocator.free(id)`. The next step is planned immediately, at the same virtual
instant. With nothing waiting and nothing holding memory, `run` blocks (on an
`asyncio.Event`, for example) until `submit` wakes it; it never calls an empty step,
sleeps, or polls.

**Events at one instant.** Requests that arrive at the same instant are submitted in
arrival order, and all of them are waiting before `run` resumes. A request that
arrives during a step, or at exactly the instant a step ends, is waiting when the
next step is planned (arrivals are delivered before the step's completion).

**The allocator is checked at every step.** At the moment `sim_engine.step` is
called, for every sequence that holds memory, `allocator.block_table(id)` must have
exactly $\lceil T / \texttt{block\_size} \rceil$ entries, where $T$ is
`kv_tokens(id) + remaining_prefill(id)` before the step, plus 1 if the sequence
decodes in it; every reference count must equal the number of those tables that
contain the block; and `free_blocks()` plus the number of distinct blocks in those
tables must be `n_blocks`. A finished sequence that was not freed is a leak and
fails this check. At most `max_running` sequences may hold memory.

In stage 2 (and 3) the pool is large enough that an `append` never fails.

### Example 2

`max_running = 2`. Requests as (arrival ms, prompt tokens, output tokens): request 0
`(0, 8, 4)`, request 1 `(0, 16, 2)`, request 2 `(5, 8, 3)`, request 3 `(45, 4, 2)`,
request 4 `(50, 12, 1)`, request 5 `(60, 4, 2)`.

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

Request 2 arrives at 5 and joins at 24, the moment request 1 frees a slot, next to
the decode of request 0. Request 3 arrives at exactly 45, the instant a step ends,
and is in the next step.

## Stage 3: chunked prefill

A step may process at most `max_batch_tokens` tokens: prompt tokens plus decoded
sequences. Planning a step becomes:

1. **Decodes first**, as in stage 2. Each costs one token of the budget.
   (`max_running <= max_batch_tokens`, so every ready sequence always decodes.)
2. **Prompts in progress.** For every sequence that holds memory and has prefill
   left, in admission order, while budget is left: a chunk of

   $$
   n = \min(\text{remaining prefill},\ \text{long\_prefill\_threshold},\ \text{budget left})
   $$

   tokens (without the threshold term when `long_prefill_threshold` is `None`).
3. **Admission**, while budget is left, fewer than `max_running` sequences hold
   memory, and a request is waiting: admit the oldest waiting request as in stage 2
   (it takes the blocks of its whole prompt now), and give it a chunk by the same
   formula.

`prefill` lists the chunks of (2), then those of (3), in that order. A prompt that
does not fit is split and continues in later steps; nobody is skipped; a sequence
whose last chunk is in this step decodes from the next step on. With a budget that
never binds and no threshold this is exactly stage 2.

### Example 3

`max_batch_tokens = 32`, `long_prefill_threshold = 24`. Request 0 `(0, 16, 4)`,
request 1 `(0, 40, 2)`, request 2 `(3, 8, 2)`.

| starts | prefill | decode | takes |
|---|---|---|---|
| 0 | `(0, 16), (1, 16)` | | 8 + 8 = 16 |
| 16 | `(1, 24), (2, 7)` | `0` | 8 + 7.75 + 1 = 16.75 |
| 32.75 | `(2, 1)` | `0, 1` | 10.25 |
| 43 | | `0, 2` | 10 |

At 0, request 1 gets the 16 tokens request 0 left. At 16 the decode of request 0
costs one token; request 1 is capped by the threshold, and request 2 gets the last
7. All 40 tokens' worth of blocks of request 1 were taken at time 0.

## Stage 4: priorities and preemption

Now the pool can run out, and requests have a `priority`: an int, **smaller is more
urgent**.

**Rank.** The rank of a request is the pair `(priority, arrival number)`, where the
arrival number counts `submit` calls from 0. Ranks are compared as tuples; a smaller
rank is better. One order governs everything: the waiting queue, the sequences
that hold memory, and both lists of a step are kept in rank order, and a preempted
sequence returns to the waiting queue with the rank it always had. (With equal
priorities, rank order is arrival order, which is the admission order of stages 2
and 3.)

Planning a step:

1. **Memory for the decodes, best rank first.** Walk through the sequences that
   hold memory, in rank order; those whose prefill is incomplete are skipped (they
   took their blocks at admission). A ready sequence needs one new block exactly
   when `kv_tokens(id) % block_size == 0`. If it needs one and
   `allocator.free_blocks()` is 0, **preempt the sequence of the worst rank that
   holds memory**: `allocator.free(victim)`, `sim_engine.preempt(victim)`, and the
   victim joins the waiting queue. Repeat until a block is free or the sequence
   that needs the block was itself the victim (it had the worst rank). If it still
   holds memory, `allocator.append(id, 1)` and it decodes in this step.
2. **Prompts in progress**, as in stage 3, in rank order.
3. **Admission, only if (1) preempted nothing**, from the head of the waiting queue
   (best rank), while budget is left and fewer than `max_running` sequences hold
   memory. The head's prefill is its `prompt_tokens`, or `remaining_prefill(id)` if
   it was preempted, and needs $\lceil \text{prefill} / \texttt{block\_size} \rceil$
   blocks. It is admitted if

   `allocator.free_blocks() - needed >= watermark_blocks`,

   **or**, when no sequence holds memory at that moment, if
   `allocator.free_blocks() - needed >= 0`. Admitting takes all of those blocks
   (`allocator.append(id, prefill)`), calls `sim_engine.admit` for a request that
   was never admitted, and gives the sequence its first chunk. If the head is not
   admitted, admission stops: nobody overtakes it.

A preempted sequence keeps its tokens: the engine continues its numbering, and the
tokens prefilled again are counted in `sim_engine.counters.recomputed_tokens`. A
victim may be a sequence that is still prefilling. Several sequences can be
preempted in one step.

### Example 4

Seven blocks of 4 tokens, `max_running = 3`, `watermark_blocks = 0`. Requests as
(arrival, prompt, output, priority): request 0 `(0, 8, 6, 1)`, request 1
`(0, 6, 7, 0)`, request 2 `(2, 7, 5, 2)`, request 3 `(30, 4, 3, 0)`.

| starts | prefill | decode | free blocks during the step | notes |
|---|---|---|---|---|
| 0 | `(1, 6), (0, 8)` | | 3 | request 1 is more urgent: first in both lists |
| 11.5 | `(2, 7)` | `1, 0` | 0 | request 0 (8 tokens) takes a third block; request 2 takes the last two |
| 23.25 | | `1, 0, 2` | 0 | |
| 34.25 | | `1, 0` | 1 | request 1 (8 tokens) needs a block and none is free: request 2, the worst rank, is preempted after 2 tokens. No admission in this step |
| 44.25 | `(3, 4)` | `1, 0` | 0 | request 3 (priority 0) is ahead of request 2 in the queue and takes the free block |
| 55.25 | | `1, 3` | 2 | request 3 (4 tokens) needs a block. The newest sequence is request 3 itself, but the worst rank is request 0 (priority 1): preempted after 5 of its 6 tokens |
| 65.25 | | `1, 3` | 2 | request 0 would need 4 blocks for its 13 tokens |
| 75.25 | `(0, 13), (2, 9)` | | 0 | requests 1 and 3 have finished. Request 0 produces its last token at 88.75 |
| 88.75 | | `2` | 4 | |
| 97.75 | | `2` | 4 | request 2 finishes at 106.75 |

22 tokens are recomputed. Request 2 produced tokens 0 and 1 before 34.25 and token
2 at 88.75: it was not started over.

## Stage 5: prefix sharing

With `prefix_sharing=True`, a new request reuses KV blocks that a sequence in
memory has already computed for the same prompt prefix. `request.tokens` is the
prompt as a tuple of token ids (`None` for a request that takes no part in
sharing).

**`fork(parent_id, child_id, n_tokens)`.** The child gets the parent's first
`n_tokens` tokens: its token count is `n_tokens` and its table is the first
$\lceil \texttt{n\_tokens} / \texttt{block\_size} \rceil$ entries of the parent's,
each of which gains one reference. `n_tokens=None` means all of the parent's tokens
(stage 1). After the checks of stage 1, raises `ValueError` if `n_tokens` is not in
`[1, parent's token count]`. Copy on write is unchanged: a child whose last shared
block it uses only partly copies that block at its first `append`.

**At admission** of a request `r` that was never admitted and has `tokens`, look at
every sequence `s` that holds memory at that moment and has `tokens`:

$$
h(r, s) = \left\lfloor \frac{\min\bigl(\text{common}(r, s),\ \text{filled}(s),\ \texttt{r.prompt\_tokens} - 1\bigr)}{\texttt{block\_size}} \right\rfloor
$$

where common is the length of the longest common prefix of the two token tuples,
and $\text{filled}(s) = \min(\texttt{kv\_tokens}(s),\ \texttt{s.prompt\_tokens})$ is
the number of prompt tokens `s` has in the KV cache now. So only **full blocks**
that `s` has **already prefilled** count, and at least one token of `r` is left to
prefill: its forward pass produces the first output token. The **donor** is the
`s` with the largest $h$; among equals, the best rank. If $h > 0$:

- `allocator.fork(donor, r.id, h * block_size)`;
- `sim_engine.admit(r, cached_tokens=h * block_size)`: the engine then counts those
  tokens as present and only `prompt_tokens - h * block_size` remain to prefill;
- `allocator.append(r.id, prompt_tokens - h * block_size)` for the rest, so the
  admission needs $\lceil \texttt{prompt\_tokens} / \texttt{block\_size} \rceil - h$
  blocks in the rule of stage 4.

Everything else is unchanged. Consequences worth knowing: two requests admitted in
the same step share nothing with each other; a sequence that was preempted returns
without sharing (its whole context is recomputed), but can be a donor again once it
holds memory; a donor can be a sequence that itself shares those blocks, so blocks
outlive the sequence that first computed them; and preempting or finishing a
sequence returns only the blocks nobody else holds.

### Example 5

Ten blocks of 4 tokens, `max_running = 4`, `prefix_sharing=True`. `S` is a system
prompt of 8 tokens, `(1, 2, ..., 8)`. Request 0 arrives at 0 with `S + (11, 12, 13)`
and 4 output tokens; request 1 at 0 with `S + (21, 22)` and 3; request 2 at 5 with
`S + (31, ..., 35)` and 2; request 3 at 30 with `(1, 2, 3, 4, 9, 9, 9, 9, 9)` and
2; request 4 at 40 with `S + (41,)` and 2.

| starts | prefill | decode | notes |
|---|---|---|---|
| 0 | `(0, 11), (1, 10)` | | admitted together: nothing to share yet |
| 13.25 | `(2, 5)` | `0, 1` | requests 0 and 1 both hold `S` in two full blocks; the donor is request 0 (best rank). Request 2 shares 8 tokens and takes 2 new blocks |
| 24.5 | | `0, 1, 2` | |
| 35.5 | `(3, 5)` | `0` | requests 1 and 2 finished. Request 3 agrees with request 0 on 4 tokens: one block |
| 45.75 | `(4, 5)` | `3` | request 0 finished; its first block lives on in request 3, which is the donor for request 4 (4 tokens in common) |
| 56 | | `4` | request 4 finishes at 65 |

36 tokens are prefilled instead of 52, and at most 9 blocks are in use. Without
sharing, the same workload runs out of memory at 26.5 and preempts request 2.

## Constraints

- `1 <= block_size <= 64`, `1 <= n_blocks <= 4096`, `1 <= max_running <= 16`,
  `max_running <= max_batch_tokens`, `0 <= watermark_blocks < n_blocks`. For every
  request, `prompt_tokens + output_tokens - 1` tokens fit in the pool, so a sequence
  running alone always finishes. Prompts have 1 to 4,000 tokens. Arrival times are
  multiples of 0.25 ms, so every time in a test is an exact float.
- **Stage 1.** Return values, block tables, reference counts, and `free_blocks()`
  are compared exactly after every call, on scripts and on random workloads, and
  the invariants of stage 1 are checked independently.
- **Stages 2 to 5.** The rules determine every step, admission, and preemption.
  The tests compare them exactly: with the literal schedules above and others like
  them, and with the reference's trace on seeded bursty and Poisson workloads of up
  to 120 requests. They also check that every request finishes with each token
  produced exactly once, in order; that no sequence is released; and the allocator
  conditions of stage 2 at every step, and that the pool is entirely free at the end.
- **Stage 3, stall bound.** Six sequences are generating when a 4,000-token prompt
  arrives, followed by three 24-token prompts, with `max_batch_tokens = 512`: the
  longest gap between two consecutive tokens of any of the six is at most **150 ms**
  (the largest step is 8 + 0.25 · 506 + 6 = 140.5 ms), with and without
  `long_prefill_threshold = 128`; with it, each short prompt has its first token
  within **120 ms** of arriving.
- **Stage 4, cost.** 120 Poisson requests in three priorities (prompts of 16 to 160
  tokens, outputs of 8 to 96), 40 blocks of 16 tokens, `max_running = 8`,
  `max_batch_tokens = 128`, `watermark_blocks = 4`: at most **3000** recomputed
  tokens and **30** preemptions. The reference recomputes 1365 tokens in 11
  preemptions; without the watermark it is several times as many.
- **Stage 5, cost.** 120 requests over three shared prefixes of 64 to 160 tokens,
  96 blocks of 16 tokens, `max_running = 8`: at most **6000** prefilled tokens
  (`sim_engine.counters.prefill_tokens`; the reference needs 4767, and 18655 without
  sharing), at most **64** blocks in use at any step (reference 54; 96 without
  sharing), and no preemption.
- Use `asyncio`, the allocator, and `sim_engine` only. Time is virtual: the real
  clock and real sleeping have no meaning here.

## Why this matters

This is the loop at the center of vLLM, TGI, SGLang, and llama.cpp's server, and
each stage is a decision those engines make before every forward pass. Continuous
batching keeps the GPU full; the token budget bounds the length of a step, and with
it the time between two streamed tokens; paging lets the engine admit optimistically
instead of reserving the longest possible output for everyone, and preemption is
the price it pays when the optimism fails; reference-counted blocks let a thousand
requests with one system prompt store it once. The parts are simple, and the
failures come from their seams: a block leaked on a path that only runs under memory
pressure, a slot that idles one step in every finish, a prompt that starves the
stream, a pool that thrashes because admission took the last free block, a shared
block freed while another request was still attending to it.
