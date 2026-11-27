## The idea

**Prefill-decode interference.** In a continuous-batching step, newly admitted prompts and active decode sequences share one forward pass, and the step runs until all of them finish. Under this model, a step that prefills $P$ tokens and decodes $D$ sequences takes $8 + 0.25\,P + D$ ms. Eight sequences decoding alone tick every 16 ms, but admitting one 3000-token document pushes that single step to 766 ms. Every active stream on the replica freezes for three-quarters of a second, raising the average time per token of a 60-token generation from 16 ms to 28 ms. Time to first token (TTFT) suffers in the opposite direction when a short prompt arrives right behind a long one and has to wait for the next step.

**Disaggregation.** Splitting the cluster gives each phase dedicated replicas. A prefill replica processes prompts and hands each request off after producing its first token. A decode replica receives the KV cache and generates the remaining tokens. Because decode steps never include long prefills, token generation stays smooth. The two pools can also be sized independently. Prefill capacity scales with prompt tokens per second and the TTFT target, while decode capacity scales with concurrent streams and the TPOT target.

**The cost of KV transfer.** Disaggregation requires shipping the prompt's full KV cache—tens of megabytes to gigabytes—across the network between the first and second token. That transfer adds nothing to TTFT because the first token is already produced on the prefill replica, but its full duration lands in the gap before the second token. Amortized over a long output, that pause is small. For a short answer, it can consume most of the TPOT budget.

**Why short prompts stay local.** A 50-token prompt adds only 12.5 ms of prefill time on a decode replica, which is less than one decode step. Sending it to the prefill pool instead puts it in a batch with long documents, delaying its first token until that heavy step finishes. In the benchmark workload, disaggregating every request drops TTFT attainment from 96.0% to 40.4% for this exact reason. Running short prompts locally on decode replicas keeps their interference negligible while avoiding the prefill-pool detour. `local_threshold` marks the crossover where prefill interference outweighs transfer and queuing overhead.

**When to pick the decode replica.** Never pick the decode replica at request arrival. Between arrival and the end of prefill lie the prefill queue and the prefill step itself—often hundreds of milliseconds during which decode loads change completely. A decision made at arrival is stale by the time the transfer starts, and reserving a decode slot from arrival leaves that replica underutilized while the request is still in the prefill pool. Choosing the decode replica right when prefill returns uses fresh load data. It leaves only one blind spot: a request currently in transfer is not yet in any replica's queue. When multiple prompts finish prefill in the same step, they all see the same least-loaded decode replica unless the coordinator tracks its own in-flight transfers. This is the same local accounting pattern used in `least-loaded-balancer`.

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

The prefill pool routes by `outstanding_tokens` because prefill latency scales with token count—one 4000-token prompt is ten times the work of four 100-token prompts. The decode pool routes by request count (`queue_len + inbound`) because a decode replica's scarce resource is its concurrent sequence slots.

## Does it pay off?

Here are the benchmark results from the statement (four replicas at 8 requests per second with 10% long documents):

| cluster | TTFT attainment | TPOT attainment |
|---|---|---|
| colocated | 95.4% | 91.8% |
| 1 prefill + 3 decode, threshold 256 | 96.0% | 98.6% |
| every request disaggregated | 40.4% | 95.2% |
| every request local to the 3 decode replicas | 90.0% | 83.2% |

The gain appears in TPOT because decode streams are no longer stalled by long prefills. TTFT stays nearly unchanged because the cluster executes the same total prefill work, now concentrated on one replica instead of four. The final row acts as a control. Three colocated replicas perform worse than four, confirming that the win comes from separating phases rather than from routing alone.

Disaggregation is not universally better. With too few prefill replicas for the prompt rate, the prefill queue backs up and TTFT collapses while decode replicas sit idle. With a slow network, KV transfer eats the TPOT budget of short answers. Under light load, colocated replicas already meet both objectives and disaggregation only adds an extra hop. The ratio of prefill to decode replicas must match the workload's ratio of prompt tokens to output tokens.

## Common mistakes

- **Disaggregating every request.** Short prompts queue behind long documents in the prefill pool and pay transfer overhead for no benefit.
- **Disaggregating nothing, or overflowing long prompts to decode replicas without bound.** Sending long prompts to decode replicas when the prefill pool looks busy brings prefill-decode interference back.
- **Using `<=` at the threshold.** Requests with `prompt_tokens == local_threshold` must be disaggregated, not served locally.
- **Choosing the decode replica at arrival.** Whether or not you reserve its slot in `inbound`, arrival-time load is stale by the time prefill finishes.
- **Ignoring in-flight transfers, or forgetting to decrement `inbound`.** Without tracking transfers, simultaneous prefill completions pile onto the same decode replica. Forgetting to decrement `inbound` makes each replica look permanently busier by its total lifetime count, degenerating into round-robin by history.
- **Submitting decode before the transfer finishes.** The simulated cluster accepts overlapping calls just as a real server might accept a request whose KV cache has not arrived and then stall or recompute, but the tests reject out-of-order execution.
- **Routing prefill by request count or decode by token count.** Each pool needs the metric matching its bottleneck.

## Complexity

Each request performs one scan of the prefill pool (if disaggregated) and one scan of the decode pool. The coordinator stores a single integer counter per decode replica.

## In the wild

- **DistServe** ("DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI 2024, arXiv:2401.09670) introduced prefill-decode disaggregation and evaluated it by *goodput*, the request rate served while meeting both TTFT and TPOT targets. It assigns the two phases to separate GPUs to eliminate interference, tunes resources and parallelism independently for each phase, and places phases according to cluster bandwidth to limit KV transfer cost.
- **vLLM** provides experimental disaggregated prefilling with separate prefill and decode instances connected by a KV-transfer connector (such as NIXL-based connectors) for moving KV caches across instances.
- **NVIDIA Dynamo** applies conditional disaggregation per request. A prompt goes to a remote prefill worker only if its uncached length (excluding prefix-cache hits) exceeds a threshold and the remote prefill queue is short; otherwise the decode worker prefills locally. The length check matches `local_threshold` here, while the queue check (omitted in this problem) protects TTFT when the prefill pool falls behind. Dynamo also chooses the decode worker first so it can pre-allocate KV blocks before queuing the remote prefill, whereas this problem uses the simpler coordinator-first flow (prefill, then choose decode).
- The lower-overhead alternative on a single pool is chunked prefill (`chunked-prefill-scheduling`), which caps prompt tokens per step to bound decode stalls without transferring KV caches.
