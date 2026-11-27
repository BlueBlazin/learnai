Implement the placement and eviction policy for a tiered KV cache across GPU memory, host memory, and disk. When GPU memory fills up, your policy demotes prefix blocks to colder tiers instead of discarding them. When a later request needs cached blocks, your policy decides whether to load them back or recompute them using a provided block store and simulated clock.

## Specification

```python
class TieredKV:
    def __init__(self, store, block_size): ...
    async def serve(self, tokens): ...  # -> dict
    def discard(self, tokens): ...  # -> bool
```

**Blocks and keys.** Let $B = \texttt{block\_size}$. As in `block-hash-prefix-cache`, block $k$ of a prompt is `tokens[k*B : (k+1)*B]`. Only full blocks of length $B$ count, and trailing tokens after the last full block are ignored. Two blocks are identical if and only if they have the same index $k$ and their prompts agree on all tokens up through the end of block $k$. Choose any hashable key per block that preserves this identity (such as a chained hash or the prefix tuple itself).

**The store.** The provided `store` manages three tiers ordered from hot to cold: `"gpu"`, `"host"`, and `"disk"`. Each stored block resides in exactly one tier. The store executes operations and keeps no policy state, so `TieredKV` must track where each block lives and how recently it was used.

| member | effect |
|---|---|
| `store.capacity` | Dict of tier capacities in blocks, e.g. `{"gpu": 4, "host": 2, "disk": 4}`. |
| `store.costs` | Dict of operation times in seconds: `compute_s`, `host_base_s`, `host_block_s`, `disk_base_s`, `disk_block_s`. |
| `await store.compute(key, parent, tokens)` | Computes one block on the GPU in `compute_s` seconds. `tokens` are the block's $B$ tokens, and `parent` is the key of block $k - 1$ (`None` for block 0), which must already be on the GPU. `key` must not be stored anywhere, and the GPU must have a free slot. |
| `await store.load(keys)` | Transfers $n \ge 1$ blocks to the GPU in one batch, all from `"host"` or all from `"disk"`, taking `base_s + block_s * n` seconds for that tier. Needs $n$ free GPU slots. Each block's parent must already be on the GPU or appear earlier in `keys`. |
| `store.demote(key, tier)` | Instantly moves a block to a colder tier (`"gpu"` to `"host"`, `"gpu"` to `"disk"`, or `"host"` to `"disk"`). Needs a free slot in `tier`. |
| `store.drop(key)` | Instantly deletes a stored block from its current tier. |

Loads and demotions move `store.block_bytes` bytes per block. Computing and dropping move zero bytes. Any invalid store call raises an error and is recorded by the harness, which fails the test even if you catch the exception. Only one `compute` or `load` may be in flight at a time.

**Pinning, victims, and placement.** While `serve(tokens)` runs, all full blocks of `tokens` are *pinned* in every tier for the duration of the call.

- **Victim.** The victim of a tier is the unpinned block in that tier that was used least recently (by the earliest `serve` call). Among blocks last used by the same `serve` call, pick the **deepest** block (largest block index). A tier whose blocks are all pinned has no victim.
- **Place** a block $b$ into a target tier:
  - If the tier has a free slot, call `demote(b, tier)`.
  - Otherwise, if the tier has a victim $v$, first place $v$ into the next tier down, then call `demote(b, tier)` into the freed slot.
  - Otherwise (the tier is full and every block in it is pinned), place $b$ into the next tier down instead.
  - Placing a block below `"disk"` calls `drop` on it.

**`await serve(tokens)`** brings every full block of `tokens` onto the GPU and returns a dict of counts. Let the prompt have $m$ full blocks, indexed $k = 0, \ldots, m - 1$. If $m = 0$, perform no store operations and return `{"gpu": 0, "host": 0, "disk": 0, "computed": 0}`. Otherwise, execute store operations in this exact order:

1. **Find the stored prefix.** Let $p \in \{0, \ldots, m\}$ be the number of **leading** blocks currently stored in any tier, so blocks $0, \ldots, p - 1$ are stored and either $p = m$ or block $p$ is missing. Any stored block behind a missing block is unreachable.
2. **Choose the cut.** Choose $c \in \{0, \ldots, p\}$. Blocks $0, \ldots, c - 1$ are *reused* (kept on the GPU or loaded), and blocks $c, \ldots, m - 1$ are *computed*. Let $h$ and $d$ be the counts of reused blocks on `"host"` and `"disk"`. Using $[x > 0]$ to denote 1 when $x > 0$ and 0 otherwise, the time cost of cut $c$ is

   $$
   t(c) = [h > 0]\,(\texttt{host\_base\_s} + h \cdot \texttt{host\_block\_s}) + [d > 0]\,(\texttt{disk\_base\_s} + d \cdot \texttt{disk\_block\_s}) + (m - c) \cdot \texttt{compute\_s}.
   $$

   Pick the $c$ with the smallest $t(c)$, breaking ties in favor of the **largest** $c$.
3. **Drop recomputed blocks.** For every block with index $k \ge c$ (in increasing index order) that is currently stored in any tier, call `store.drop` on it because it will be recomputed.
4. **Make room on the GPU.** Let $g$ be the number of reused blocks already on `"gpu"`. The call needs $m - g$ free GPU slots for the blocks it will load or compute. While `"gpu"` has fewer than $m - g$ free slots, select the GPU's victim and place it in `"host"`.
5. **Load.** If $h > 0$, call `await store.load` once with the reused `"host"` blocks in increasing index order. Then, if $d > 0$, call `await store.load` once with the reused `"disk"` blocks in increasing index order.
6. **Compute.** For $k = c, \ldots, m - 1$ in order, call `await store.compute` for block $k$.
7. **Update recency and return.** Mark all $m$ blocks as used by this `serve` call, and return `{"gpu": g, "host": h, "disk": d, "computed": m - c}`.

**`discard(tokens)`** removes one block from the store (for example, upon expiry). It drops the **last** full block of `tokens` from whichever tier holds it and returns `True`. If that block is not stored or `tokens` has no full block, it does nothing and returns `False`. Any blocks behind a discarded block remain in place, unreachable, until they are dropped as victims or in step 3.

`serve` is never called while another `serve` is running. `tokens` is a list of ints and must not be modified. Use `asyncio` only through the store's two awaitable calls, which are where simulated time advances.

> [!NOTE]
> Breaking ties deepest-first keeps every chain ordered by tier. A block is never in a hotter tier than its parent, so the reused prefix of any prompt consists of GPU blocks, then host blocks, then disk blocks. Loading the host run before the disk run therefore always finds every parent already on the GPU.

## Example

Suppose `block_size = 2`, capacity is `{"gpu": 4, "host": 2, "disk": 4}`, and costs are `compute_s = 0.25`, host `0.125 + 0.0625 n`, and disk `0.5 + 0.125 n`. Prompts `A` and `B` each have three full blocks: `A0 A1 A2` and `B0 B1 B2`.

| call | starts | store operations, in order | returns | ends |
|---|---|---|---|---|
| `serve(A)` | 0 | compute `A0`, `A1`, `A2` | `computed: 3` | 0.75 |
| `serve(B)` | 1 | demote `A2` to host, `A1` to host; compute `B0`, `B1`, `B2` | `computed: 3` | 1.75 |
| `serve(A)` | 2 | demote `B2` to disk, `B1` to disk; load `[A1, A2]` from host | `gpu: 1, host: 2` | 2.25 |
| `serve(B)` | 3 | drop `B1`, `B2`; demote `A2` to host, `A1` to host; compute `B1`, `B2` | `gpu: 1, computed: 2` | 3.5 |

(Zero-valued counts are omitted from the table for readability; the returned dict always contains all four keys.)

- **Second call.** Three blocks must arrive on the GPU and only one slot is free. All blocks of `A` were last used by the first call, so the victims are chosen deepest-first: `A2`, then `A1`. `A0` stays on the GPU.
- **Third call.** `A0` is on the GPU while `A1` and `A2` are on the host. Loading both from host takes $0.125 + 2 \cdot 0.0625 = 0.25$ s, whereas computing them takes $0.5$ s, so $c = 3$. Two GPU slots are needed. The first GPU victim is `B2`, but the host tier is full and both of its blocks (`A1`, `A2`) are pinned by this call. `B2` therefore passes through to disk, followed by `B1`. One batched transfer then loads `A1` and `A2`.
- **Fourth call.** `B0` is on the GPU while `B1` and `B2` are on disk. Loading two blocks from disk takes $0.5 + 0.25 = 0.75$ s, whereas computing them takes $0.5$ s. The cut is therefore $c = 1$. Both disk copies are dropped, room is made on the GPU by demoting `A2` and `A1` to host, and `B1` and `B2` are computed.

In total, 8 blocks cross a tier boundary (2, then 4, then 2), and the four calls spend $0.75 + 0.75 + 0.25 + 0.5 = 2.25$ s of store time.

## Constraints

- $1 \le \texttt{block\_size} \le 16$, tier capacities are between 1 and 16 blocks, a prompt never has more full blocks than the GPU has slots, and each test runs at most a few hundred calls.
- All costs are positive multiples of $2^{-6}$ s, so every sum is exact in floating point and times are compared with `==`. Simultaneous events do not occur because only one call runs at a time.
- For every call, tests compare your policy against the reference on the exact sequence of store operations, the returned dict, and the virtual end time. Any store error fails the test.
- **Cost.** Two totals are tracked and must not exceed the reference policy's values: `bytes_moved` (total bytes loaded or demoted) and `serve_time` (total seconds spent in loads and computations).
- Do not read the real wall clock or call `sleep`.

## Why this matters

GPU memory is mostly consumed by active requests, leaving little room for cached prefixes such as system prompts, multi-turn conversations, or shared documents. A single-tier cache must evict those blocks under pressure and lose them. Tiered KV stores offload them to larger, cheaper host memory and local disk so returning requests can reload prefixes instead of repeating prefill.

Getting the policy right matters as much as having the tiers. Transfers have fixed latency and finite bandwidth, so recomputing beats loading for short prefixes on slow tiers. A policy that always loads makes short requests slower than having no cache at all. Eviction details are equally fragile. Demoting a parent before its children strands cold chains without their heads. Forgetting to pin the active request's blocks causes the store to demote the very blocks it is about to load.
