Build the policy of a tiered KV store: prefix blocks live on the GPU, in host
memory, or on disk, each tier larger and slower than the one above. Memory pressure
pushes blocks down instead of throwing them away, and a request that needs them
brings them back up, unless computing them again is faster than the transfer. The
store that holds the blocks is given; you write the index that decides what moves
where, on a simulated clock that charges for every load and every computation.

## Specification

```python
class TieredKV:
    def __init__(self, store, block_size): ...
    async def serve(self, tokens): ...  # -> dict
    def discard(self, tokens): ...  # -> bool
```

**Blocks and keys**, as in `block-hash-prefix-cache`. With $B$ = `block_size`, block
$k$ of a prompt is `tokens[k*B : (k+1)*B]`; only full blocks count, and the tokens
after the last full block are ignored. Two blocks are the same block exactly when
they have the same index and their prompts agree on all tokens up to the end of the
block. You choose a hashable **key** for every block that respects this identity
(the chained hash of the prerequisite, or the prefix itself).

**The store** has three tiers, `"gpu"`, `"host"`, and `"disk"`, from hot to cold. A
stored block is in exactly one tier. The store does what it is told and keeps no
policy; your class keeps its own record of where each block is and how recently it
was used.

| member | effect |
|---|---|
| `store.capacity` | dict: slots per tier, in blocks, e.g. `{"gpu": 4, "host": 2, "disk": 4}` |
| `store.costs` | dict of seconds: `compute_s`, `host_base_s`, `host_block_s`, `disk_base_s`, `disk_block_s` |
| `await store.compute(key, parent, tokens)` | Computes one block on the GPU; takes `compute_s`. `tokens` are the block's own $B$ tokens; `parent` is the key of the block before it (`None` for block 0) and must be on the GPU. `key` must not be stored anywhere. Needs a free GPU slot. |
| `await store.load(keys)` | One transfer of $n \ge 1$ blocks to the GPU, all from the host tier or all from the disk tier; takes `base_s + block_s * n` of that tier. Needs $n$ free GPU slots. Each block's parent must be on the GPU or earlier in `keys`. |
| `store.demote(key, tier)` | Moves a block down, GPU to `"host"`, GPU to `"disk"`, or host to `"disk"`. Instant. Needs a free slot in `tier`. |
| `store.drop(key)` | Deletes a stored block. Instant. |

Loads and demotions move `store.block_bytes` bytes per block; computing and dropping
move none. A call the store cannot carry out raises an error and is recorded; a
solution that causes one fails, whether or not it catches the error. Only one
`compute` or `load` may be in flight at a time.

**`await serve(tokens)`** brings every full block of the prompt onto the GPU and
returns what it did. Let the prompt have $m$ full blocks, $k = 0, \ldots, m - 1$.
The store operations of one call are made in exactly this order:

1. **Find.** $p$ is the number of **leading** blocks that are stored in some tier:
   blocks $0, \ldots, p - 1$ are stored and block $p$ is not (or $p = m$). A stored
   block behind a missing one is not reachable.
2. **Choose the cut** $c \in \{0, \ldots, p\}$: blocks before $c$ are *reused* (left
   on the GPU, or loaded), blocks from $c$ on are *computed*. If the reused blocks
   include $h$ on the host tier and $d$ on the disk tier, the cut costs

   $$
   t(c) = [h > 0]\,(\texttt{host\_base\_s} + h \cdot \texttt{host\_block\_s}) + [d > 0]\,(\texttt{disk\_base\_s} + d \cdot \texttt{disk\_block\_s}) + (m - c) \cdot \texttt{compute\_s}.
   $$

   Take the $c$ with the smallest $t(c)$, and the **largest** such $c$ on a tie.
3. **Drop** every stored block with index $\ge c$, in index order, whatever tier it
   is in: it will be computed again.
4. **Make room.** The blocks that are about to arrive on the GPU are the $m$ blocks
   minus the reused ones already there. While the GPU has fewer free slots than
   that, take the GPU's *victim* and *place* it in the host tier (both defined
   below).
5. **Load.** If $h > 0$, one `load` of the reused host blocks, in index order. Then,
   if $d > 0$, one `load` of the reused disk blocks, in index order.
6. **Compute** blocks $c, \ldots, m - 1$, in order, one `compute` each.
7. All $m$ blocks are now *used by this call*. Return
   `{"gpu": g, "host": h, "disk": d, "computed": m - c}`, where $g$ is the number of
   reused blocks that were already on the GPU. (A prompt without a full block
   causes no operation and returns four zeros.)

**Victim.** The blocks of the prompt being served are *pinned* in every tier for
the duration of the call. The victim of a tier is, among its blocks that are not
pinned, the one used least recently, that is, by the earliest `serve` call; among
blocks last used by the same call, the **deepest** (largest index). A tier whose
blocks are all pinned has no victim.

**Place** a block $b$ in a tier:

- If the tier has a free slot, `demote` $b$ there.
- Otherwise, if the tier has a victim $v$: first place $v$ in the next tier down,
  then `demote` $b$ into the slot.
- Otherwise (full, and everything in it pinned): place $b$ in the next tier down
  instead.
- Placing a block below the disk tier means `drop` it.

**`discard(tokens)`** removes one block from the store, for example because it
expired: the **last** full block of `tokens` is dropped from whatever tier holds it,
and the method returns `True`. If that block is not stored, or `tokens` has no full
block, nothing happens and it returns `False`. Blocks behind a discarded block stay
where they are, unreachable, until they are dropped as victims or in step 3.

`serve` is never called while another `serve` is running. `tokens` is a list of
ints; do not modify it. Use `asyncio` only through the store: its two awaitable
calls are where time passes.

> [!NOTE]
> Deepest-first is what keeps a chain ordered: a block is never in a hotter tier
> than its parent, so the reused blocks of a prompt are GPU blocks, then host
> blocks, then disk blocks, and loading the host run before the disk run always
> finds every parent in place.

## Example

`block_size = 2`, capacity `{"gpu": 4, "host": 2, "disk": 4}`, and costs
`compute_s = 0.25`, host `0.125 + 0.0625 n`, disk `0.5 + 0.125 n`. `A` and `B` are
prompts of three blocks each, `A0 A1 A2` and `B0 B1 B2`.

| call | starts | store operations, in order | returns | ends |
|---|---|---|---|---|
| `serve(A)` | 0 | compute `A0`, `A1`, `A2` | `computed: 3` | 0.75 |
| `serve(B)` | 1 | demote `A2` to host, `A1` to host; compute `B0`, `B1`, `B2` | `computed: 3` | 1.75 |
| `serve(A)` | 2 | demote `B2` to disk, `B1` to disk; load `[A1, A2]` from host | `gpu: 1, host: 2` | 2.25 |
| `serve(B)` | 3 | drop `B1`, `B2`; demote `A2` to host, `A1` to host; compute `B1`, `B2` | `gpu: 1, computed: 2` | 3.5 |

(Counts that are 0 are left out of the table; the dict always has all four keys.)

- **Second call.** Three blocks must arrive and one slot is free. All of `A` was
  last used by the first call, so the victims are its deepest blocks first: `A2`,
  then `A1`. `A0` stays on the GPU.
- **Third call.** `A0` is on the GPU and `A1`, `A2` on the host. Loading both takes
  $0.125 + 2 \cdot 0.0625 = 0.25$ s; computing them would take $0.5$ s, so $c = 3$.
  Two slots are needed. The victim is `B2`, but the host tier is full and both its
  blocks are pinned (they belong to this prompt), so `B2` is placed one tier
  further down, and `B1` after it. Then one transfer brings `A1` and `A2` back.
- **Fourth call.** `B0` is on the GPU and `B1`, `B2` on disk. A disk transfer of two
  blocks takes $0.5 + 0.25 = 0.75$ s; computing two blocks takes $0.5$ s. So
  $c = 1$: the two disk copies are dropped, room is made, and the blocks are
  computed.

In all, 8 blocks crossed a tier boundary (2, then 4, then 2), and the four calls
took $0.75 + 0.75 + 0.25 + 0.5 = 2.25$ s of store time.

## Constraints

- $1 \le$ `block_size` $\le 16$; capacities from 1 to 16 blocks per tier; a prompt
  never has more full blocks than the GPU has slots; at most a few hundred calls.
- All costs are positive multiples of $2^{-6}$ s, so every sum is exact and times
  are compared with `==`. Simultaneous events do not occur: one call runs at a time.
- For every call the tests compare, with the reference policy run on the same
  script: the store operations of the call, in order; the returned value; and the
  virtual time at which the call ends. Any store error fails the case.
- **Cost.** Two totals are reported and must not exceed the policy's: `bytes_moved`
  (bytes loaded or demoted) and `serve_time` (seconds of loads and computations).
- Do not use the real clock or sleep yourself.

## Why this matters

GPU memory holds the KV cache of the requests that are running and little else.
Everything a prefix cache could reuse later (a long system prompt, a conversation
whose user is typing, a document someone will ask about again) competes for the
same memory, and a cache that can only evict loses it. Tiered stores keep it: host
memory is larger than GPU memory, local disks are larger again, and bringing a
prefix back is usually far cheaper than prefilling it again. Usually.
A transfer has a fixed cost and a bandwidth, a prefill has a speed, and for a short
prefix on a slow tier recomputing wins; a store that always loads makes those
requests slower than no cache at all. The bookkeeping decides the rest: demote a
parent before its children and the cold tiers fill with chains whose heads are
missing; forget to pin the blocks of the request in flight and the store pushes
down the very blocks it is about to load.
