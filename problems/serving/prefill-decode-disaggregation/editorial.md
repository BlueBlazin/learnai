## The idea

**Interference.** In a continuous-batching step, the prefill tokens of newly
admitted prompts and the decode of running sequences share one forward pass, and
the step takes as long as all of it. Here a step costs $8 + 0.25\,P + D$ ms: eight
sequences decoding alone tick every 16 ms; admit one 3000-token document and that
step takes 766 ms. Every stream on the replica freezes for three quarters of a
second, and a request that needed 60 more tokens has its average time per token
pushed from 16 to 28 ms by that single event. Time to first token suffers the same
way in the other direction: a short prompt that arrives just after a long one is
prefilled in the next step, behind it.

**Disaggregation** gives each phase its own replicas. A prefill replica runs
prompts only and hands each request off after its first token; a decode replica
receives the keys and values and continues from the second token. Decode steps
never contain a long prefill, so the token pace is steady, and the two pools can
be sized separately: how many prefill replicas is a question about prompt tokens
per second and the TTFT objective, how many decode replicas a question about
concurrent streams and the TPOT objective.

**The price** is the transfer: the whole KV cache of the prompt, tens of megabytes
to gigabytes, crosses the network between the first and the second token. It adds
nothing to TTFT (the first token is produced where the prompt was prefilled) and
adds its full duration to the gap before the second token. Averaged over a long
output that is small; for a short answer it is most of the TPOT budget.

**Not every request should pay it.** A 50-token prompt costs a decode replica
12 ms of prefill, less than one decode step. Sending it through the prefill pool
instead puts it in a batch with whatever long prompt is being prefilled there, and
its first token arrives when that batch ends. In the statement's workload,
disaggregating everything drops TTFT attainment from 96% to 40% for exactly this
reason. So short prompts are prefilled locally on the decode replica, where their
interference is negligible, and only long ones take the detour. The threshold is
where the interference a prompt would cause exceeds what the detour costs.

**When to choose the decode replica.** Not at arrival. Between arrival and the end
of the prefill lie the prefill queue and the prefill itself, often hundreds of
milliseconds, in which the decode replicas' load changes completely. A choice made
at arrival is stale when it is used, and if it is also *reserved* (counted as load
from arrival), the replica is kept half idle for a request that is not there yet.
Choosing at the end of the prefill uses current information. It leaves one blind
spot: a request in transfer is in nobody's queue. Several prompts prefilled in the
same step finish at the same instant, all see the same least loaded decode replica,
and all go there, unless the coordinator counts its own transfers in flight. This
is the same local accounting that `least-loaded-balancer` needs between load
reports.

## Reference solution (Python)

```python
class Coordinator:
    def __init__(self, cluster, prefill_ids, decode_ids, local_threshold):
        self.cluster = cluster
        self.prefill_pool = [cluster.replica(i) for i in prefill_ids]
        self.decode_pool = [cluster.replica(i) for i in decode_ids]
        self.local_threshold = local_threshold
        self.inbound = dict.fromkeys(decode_ids, 0)

    def pick_decode(self):
        return min(self.decode_pool, key=lambda r: (r.queue_len + self.inbound[r.id], r.id))

    async def serve(self, request):
        if request.prompt_tokens < self.local_threshold:
            await self.pick_decode().submit(request)
            return

        prefill = min(self.prefill_pool, key=lambda r: (r.outstanding_tokens, r.id))
        await prefill.submit(request, phase="prefill")

        decode = self.pick_decode()
        self.inbound[decode.id] += 1
        await self.cluster.transfer(request, prefill, decode)
        self.inbound[decode.id] -= 1
        await decode.submit(request, phase="decode")
```

The prefill replica is chosen by outstanding tokens because prefill work is
proportional to tokens: one 4000-token prompt is ten times the work of four
100-token prompts. The decode replica is chosen by requests because a decode
replica's scarce resource is its slots.

## Does it pay off?

The statement's table, four replicas at 8 requests per second with 10% long
documents:

| cluster | TTFT attainment | TPOT attainment |
|---|---|---|
| colocated | 95.4% | 91.8% |
| 1 prefill + 3 decode, threshold 256 | 96.0% | 98.6% |
| every request disaggregated | 40.4% | 95.2% |
| every request local to the 3 decode replicas | 90.0% | 83.2% |

The gain is in TPOT, as expected: decode is what was being interfered with. TTFT
is roughly unchanged, because the same prefill work is done either way, now on one
replica instead of four. The last row is the control: three colocated replicas are
worse than four, so the improvement comes from the separation and not from the
routing. The gain is not unconditional. With too few prefill replicas for the
prompt load, the prefill queue grows and TTFT collapses while decode replicas idle;
with a slow network, the transfer eats the TPOT budget of short answers; with
light load, colocated replicas meet both objectives and disaggregation only adds a
hop. The split between the pools has to follow the workload's ratio of prompt
tokens to output tokens.

## Common mistakes

- **Disaggregating everything.** Short prompts queue behind long ones and pay a
  transfer for nothing.
- **Disaggregating nothing**, or sending long prompts to decode replicas when the
  prefill pool looks busy, without bound: the interference is back.
- **`<=` at the threshold.**
- **Choosing the decode replica at arrival**, with or without reserving it.
- **Ignoring transfers in flight**, or never decrementing the count when they land
  (every decode replica then looks busier by the number of requests ever sent to
  it, and the choice degenerates into round robin by history).
- **Not waiting for the transfer**: starting it and submitting the decode at once.
  In the simulation the cluster accepts this, as a real decode server would accept
  a request whose cache has not arrived and then block or recompute; the tests
  check the order.
- **Choosing the prefill replica by request count**, or the decode replica by
  tokens.

## Complexity

Each request costs one scan of each pool. The coordinator holds one counter per
decode replica.

## In the wild

- **DistServe** ("DistServe: Disaggregating Prefill and Decoding for
  Goodput-optimized Large Language Model Serving", OSDI 2024, arXiv:2401.09670)
  introduced the split, measured by goodput: the request rate served while meeting
  both a TTFT and a TPOT objective. It assigns the two phases to different GPUs to
  remove prefill-decode interference, chooses the resources and parallelism of
  each phase for the application's two objectives, and places the phases according
  to the cluster's bandwidth to limit the cost of moving KV caches.
- **vLLM** documents "Disaggregated Prefilling (experimental)": a prefill instance
  and a decode instance, with a connector that transfers the KV caches between
  them (NIXL-based and other connectors). The feature is marked experimental and
  relies on those connectors for production use.
- **NVIDIA Dynamo** makes the decision per request ("conditional disaggregation"):
  a prefill is sent to a remote prefill worker only if its length, not counting a
  prefix-cache hit, exceeds a threshold, and only if the remote prefill queue is
  short; otherwise the decode worker prefills locally. The first condition is this
  problem's `local_threshold`. The second, which this problem leaves out, protects
  TTFT when the prefill pool is behind. In Dynamo the decode worker is chosen
  first and allocates the KV blocks before the remote prefill is queued; this
  problem's order (prefill, then choose) is the simpler one of a coordinator in
  front of two engine pools.
- The cheaper alternative on a single pool is chunked prefill
  (`chunked-prefill-scheduling`): cap the prompt tokens per step so that decodes
  are delayed by a bounded amount. It bounds the interference instead of removing
  it and needs no transfer.
