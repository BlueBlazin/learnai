Split a cluster into replicas that only prefill prompts and replicas that only
decode, and write the coordinator that takes each request through both: prefill in
one pool, move the KV cache, decode in the other. A long prefill then no longer
stalls every token stream that shares its replica. The transfer is not free and
neither is the detour, so the coordinator also has to decide which requests stay
local, and it has to choose the decode replica at the right moment.

## Specification

```python
class Coordinator:
    def __init__(
        self, cluster, prefill_ids: list[int], decode_ids: list[int], local_threshold: int
    ): ...
    async def serve(self, request) -> None: ...
```

`cluster` is the kit's [`SimCluster`](../../../python/src/learnai/kit/simcluster.py).
`prefill_ids` and `decode_ids` are the replica ids of the two pools (disjoint, each
non-empty, in no particular order). The test starts `serve(request)` as a task at
the request's arrival; it returns when the request has finished. Time is virtual,
in **milliseconds**.

**The cluster.** `cluster.replica(id)` is a replica: a continuous-batching server
that runs at most 8 sequences at once, first come, first served. In each step,
every sequence that is ready decodes one token and every newly admitted prompt is
prefilled whole; a step over $P$ prefilled tokens and $D$ decoded sequences takes
$8 + 0.25\,P + D$ ms, and nothing on that replica produces a token until it ends.
What the coordinator may use:

| call | effect |
|---|---|
| `replica.queue_len` | requests the replica holds, waiting or running |
| `replica.outstanding_tokens` | prompt tokens it still has to prefill plus output tokens it still has to produce |
| `await replica.submit(request)` | serve the request entirely there; returns when it has finished |
| `await replica.submit(request, phase="prefill")` | prefill only; returns when the **first output token** has been produced, and the replica keeps nothing |
| `await cluster.transfer(request, src, dst)` | move the request's KV cache; takes $\tfrac{1}{32}$ ms per prompt token |
| `await replica.submit(request, phase="decode")` | produce the remaining output tokens; returns when the request has finished |

`submit` takes effect when it is called (the request is in the replica's queue at
once); only the result is awaited.

**The policy.**

1. **Short prompts stay local.** If `request.prompt_tokens < local_threshold`, the
   request is served entirely on a decode replica, chosen as in (3), with one
   `submit`.
2. **Prefill.** Otherwise it is submitted with `phase="prefill"` to the prefill
   replica with the fewest `outstanding_tokens`; among equals, the lowest id.
3. **Choose the decode replica when it is needed.** When the prefill has returned,
   and not before, choose the decode replica with the smallest *load*, lowest id
   among equals, where

   $$\text{load}(d) = \texttt{d.queue\_len} + \text{inbound}(d)$$

   and $\text{inbound}(d)$ is the number of requests this coordinator is currently
   transferring to $d$.
4. **Transfer, then decode.** `await cluster.transfer(request, prefill_replica,
   decode_replica)`. The request counts toward $\text{inbound}(d)$ from the choice
   in (3) until the transfer has completed. Then submit it to the decode replica
   with `phase="decode"`.

Decisions (1) and (2) are made when `serve` starts, before it awaits anything;
decision (3) at the moment the prefill's `submit` returns, without awaiting
anything in between; the decode `submit` at the moment the transfer returns.

**Order at one instant.** Arrival times in the tests never coincide with the end of
a step or of a transfer, except where stated: several prompts prefilled in one step
return at the same instant, in the order they were submitted to that replica, and
their decisions (3) are made one after another in that order, each seeing the
`inbound` counts left by the one before. Requests that arrive together are served
in arrival order.

## Example

Replica 0 is the prefill pool, replicas 1 and 2 are the decode pool,
`local_threshold = 64`. Requests as (arrival, prompt tokens, output tokens):
request 0 `(0, 16, 3)`, request 1 `(0, 320, 3)`, request 2 `(2, 64, 2)`, requests 3
and 4 `(90, 320, 2)`.

| time | event |
|---|---|
| 0 | request 0 (16 tokens, below the threshold) is submitted whole to replica 1. Request 1 goes to replica 0 for prefill: $8 + 0.25 \cdot 320 = 88$ ms |
| 2 | request 2 has 64 tokens, not below the threshold: prefill on replica 0, which is busy until 88 |
| 12, 21, 30 | replica 1 produces the three tokens of request 0 |
| 88 | first token of request 1. Both decode replicas are idle: replica 1. Transfer, $320 / 32 = 10$ ms. Replica 0 starts the prefill of request 2 ($8 + 16 = 24$ ms) |
| 90 | requests 3 and 4 arrive and are submitted to replica 0 |
| 98 | the KV cache of request 1 has arrived: submitted to replica 1 for decode (tokens at 107 and 116) |
| 112 | first token of request 2. Replica 1 holds request 1, replica 2 is idle: replica 2. Transfer, 2 ms; decode from 114, finished at 123. Replica 0 prefills requests 3 and 4 in one step, $8 + 0.25 \cdot 640 = 168$ ms |
| 280 | first tokens of requests 3 and 4. Request 3 chooses replica 1 (both idle, lowest id). For request 4, replica 1 has one transfer inbound: replica 2 |
| 290 | both transfers land; both requests decode their second token, finished at 299 |

Without the inbound count, request 4 would also have chosen replica 1 at 280.

## Constraints

- Up to 8 replicas; `1 <= local_threshold`; prompts of 1 to 4000 tokens; every
  request has at least 2 output tokens. The coordinator may read
  `request.prompt_tokens` and must not read `request.output_tokens`.
- The policy determines every `submit` and every transfer. The tests compare them
  (time, request, replica) exactly, with literal values on small scenarios and
  with the reference on seeded workloads of up to 500 requests. They also check,
  per request, that a prompt below the threshold is submitted once, to a decode
  replica; that every other prompt is prefilled in the prefill pool, transferred
  exactly once from that replica to its decode replica, not before its first token,
  and submitted for decode not before the transfer has completed; and that no
  decode replica ever prefills a prompt at or above the threshold.
- **Against a colocated cluster.** 500 requests arrive at 8 per second: 90% chat
  requests (prompts of 32 to 128 tokens, outputs of 32 to 128) and 10% documents
  (prompts of 1500 to 4000 tokens, outputs of 16 to 64). A request meets the TTFT
  objective if its first token comes within $100 + 0.5 \cdot \text{prompt\_tokens}$
  ms of its arrival, and the TPOT objective if the time from its first to its last
  token, divided by `output_tokens - 1`, is at most 16 ms. With one prefill replica,
  three decode replicas, and `local_threshold = 256`, the fraction of requests that
  meet the TTFT objective must be at least that of four colocated replicas (every
  replica serves whole requests; each goes to the shortest queue), and the fraction
  that meet the TPOT objective at least 5 points higher.

  | cluster of four replicas | TTFT attainment | TPOT attainment |
  |---|---|---|
  | colocated | 95.4% | 91.8% |
  | this problem | 96.0% | 98.6% |
  | every request disaggregated | 40.4% | 95.2% |
  | pools, but every request local to the decode pool | 90.0% | 83.2% |

- Use `asyncio` and the cluster only.

## Why this matters

Prefill and decode want opposite things from a GPU. A prefill is one large,
compute-bound pass whose latency is felt once, as time to first token; decode is
thousands of small passes whose latency is felt continuously, as the pace of the
stream. On a shared replica every long prompt that is admitted makes all running
streams skip a beat for as long as its prefill takes, and the only defenses
(chunking the prefill, capping the batch) trade one objective against the other.
Separate pools remove the interference and let each side be sized and tuned for its
own objective, at the price of moving the KV cache between them. Whether that price
is worth paying depends on the request, which is why the systems that do this in
production all make it conditional.
