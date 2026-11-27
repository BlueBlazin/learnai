Split a serving cluster into separate prefill and decode pools, and implement the coordinator that routes each request through them. Prefilling long prompts on dedicated replicas prevents them from stalling active decode streams on shared hardware. Because moving a KV cache across the network has its own latency cost, your coordinator must keep short prompts local to the decode pool and pick each decode replica at the right time.

## Specification

```python
class Coordinator:
    def __init__(
        self, cluster, prefill_ids: list[int], decode_ids: list[int], local_threshold: int
    ): ...
    async def serve(self, request) -> None: ...
```

`cluster` is an instance of [`SimCluster`](../../../python/src/learnai/kit/simcluster.py). `prefill_ids` and `decode_ids` are non-empty, disjoint lists of replica IDs in no particular order. The test harness launches `serve(request)` as a task when each request arrives, and `serve` returns once that request finishes. Time is virtual and measured in **milliseconds**.

**The cluster.** Calling `cluster.replica(id)` returns a replica, which is a continuous-batching server running at most 8 sequences at once in first-come, first-served order. In each step, every ready sequence decodes one token and every newly admitted prompt is prefilled whole. A step over $P$ prefilled tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms, and no sequence on that replica produces a token until the step finishes. Your coordinator may use the following properties and methods:

| call | effect |
|---|---|
| `replica.queue_len` | Number of requests currently waiting or running on the replica |
| `replica.outstanding_tokens` | Prompt tokens still to prefill plus output tokens still to produce on the replica |
| `await replica.submit(request)` | Serve the request end-to-end on this replica and return when it finishes |
| `await replica.submit(request, phase="prefill")` | Prefill only, returning once the **first output token** is produced (the replica retains no state afterward) |
| `await cluster.transfer(request, src, dst)` | Transfer the request's KV cache from `src` to `dst`, taking $\tfrac{1}{32}$ ms per prompt token |
| `await replica.submit(request, phase="decode")` | Produce the remaining output tokens and return when the request finishes |

Calling `submit` places the request into the replica's queue immediately, before awaiting its completion.

**The policy.**

1. **Keep short prompts local.** If `request.prompt_tokens < local_threshold`, choose a decode replica using rule (3) and serve the entire request there with a single `await replica.submit(request)`.
2. **Prefill long prompts in the prefill pool.** Otherwise, submit the request with `phase="prefill"` to the prefill replica with the fewest `outstanding_tokens`. Break ties by lowest replica ID.
3. **Choose the decode replica only when needed.** Once the prefill finishes (and not before), pick the decode replica $d$ with the smallest *load*, breaking ties by lowest replica ID. Define the load of decode replica $d$ as

   $$\text{load}(d) = \texttt{d.queue\_len} + \text{inbound}(d)$$

   where $\text{inbound}(d)$ is the number of requests this coordinator is currently transferring to $d$.
4. **Transfer, then decode.** Run `await cluster.transfer(request, prefill_replica, decode_replica)`. The request counts toward $\text{inbound}(d)$ from the moment it is chosen in step (3) until the transfer completes. Once the transfer finishes, submit the request to the decode replica with `phase="decode"`.

Make decisions (1) and (2) when `serve` starts, before awaiting anything. Make decision (3) immediately when the prefill `submit` returns, with no `await` in between. Submit the decode phase immediately when `transfer` returns.

**Order at one instant.** Arrival times in the tests never coincide with the end of a step or transfer, with one exception. Multiple prompts prefilled in the same step return at the same instant in the order they were submitted to that prefill replica. Their step (3) decisions run sequentially in that order, so each request sees the updated `inbound` counts from earlier requests in the batch. Simultaneous arrivals are served in arrival order.

## Example

Replica 0 is the prefill pool, replicas 1 and 2 form the decode pool, and `local_threshold = 64`. Five requests arrive, written as `(arrival, prompt_tokens, output_tokens)`: request 0 `(0, 16, 3)`, request 1 `(0, 320, 3)`, request 2 `(2, 64, 2)`, and requests 3 and 4 `(90, 320, 2)`.

| time | event |
|---|---|
| 0 | Request 0 (16 tokens, below the threshold) is submitted whole to replica 1. Request 1 goes to replica 0 for prefill ($8 + 0.25 \cdot 320 = 88$ ms). |
| 2 | Request 2 has 64 tokens (not below the threshold), so it is sent to replica 0 for prefill and waits until 88 ms. |
| 12, 21, 30 | Replica 1 produces the three output tokens of request 0. |
| 88 | Request 1 finishes prefill and produces its first token. Both decode replicas are idle, so it picks replica 1 and starts a $320 / 32 = 10$ ms transfer. Replica 0 begins prefilling request 2 ($8 + 16 = 24$ ms). |
| 90 | Requests 3 and 4 arrive and are submitted to replica 0. |
| 98 | Request 1's KV cache arrives at replica 1 and starts decoding (producing tokens at 107 ms and 116 ms). |
| 112 | Request 2 finishes prefill. Replica 1 holds request 1 while replica 2 is idle, so request 2 picks replica 2, transfers in 2 ms, decodes from 114 ms, and finishes at 123 ms. Replica 0 prefills requests 3 and 4 together in one step ($8 + 0.25 \cdot 640 = 168$ ms). |
| 280 | Requests 3 and 4 finish prefill together. Request 3 picks replica 1 (both idle, lowest ID). Request 4 sees one inbound transfer on replica 1, so it picks replica 2. |
| 290 | Both transfers finish. Both requests decode their second token and finish at 299 ms. |

Without tracking inbound transfers, request 4 would also have picked replica 1 at 280 ms.

## Constraints

- Up to 8 replicas, `1 <= local_threshold`, prompts of 1 to 4000 tokens, and at least 2 output tokens per request. The coordinator may read `request.prompt_tokens` and must not read `request.output_tokens`.
- The policy uniquely determines every `submit` and every `transfer`. Tests compare `(time, request, replica)` exactly against literal values on small scenarios and against the reference implementation on seeded workloads of up to 500 requests. Per request, tests also verify that prompts below `local_threshold` are submitted once to a decode replica and that no decode replica ever prefills a prompt at or above `local_threshold`. Every other prompt must be prefilled in the prefill pool, transferred exactly once from that prefill replica to the chosen decode replica (not before the first token), and submitted for decode not before the transfer completes.
- **Against a colocated cluster.** In the benchmark workload, 500 requests arrive at 8 requests per second. Of these, 90% are chat requests (prompts of 32 to 128 tokens, outputs of 32 to 128 tokens) and 10% are document requests (prompts of 1500 to 4000 tokens, outputs of 16 to 64 tokens). A request meets the time-to-first-token (TTFT) objective if its first token arrives within $100 + 0.5 \cdot \text{prompt\_tokens}$ ms of arrival. It meets the time-per-output-token (TPOT) objective if the time from its first to its last token, divided by `output_tokens - 1`, is at most 16 ms. With one prefill replica, three decode replicas, and `local_threshold = 256`, the fraction of requests meeting the TTFT objective must be at least that of four colocated replicas (where every replica serves whole requests and each request goes to the shortest queue), and the fraction meeting the TPOT objective must be at least 5 percentage points higher.

  | cluster of four replicas | TTFT attainment | TPOT attainment |
  |---|---|---|
  | colocated | 95.4% | 91.8% |
  | this problem | 96.0% | 98.6% |
  | every request disaggregated | 40.4% | 95.2% |
  | pools, but every request local to the decode pool | 90.0% | 83.2% |

- Use `asyncio` and the cluster only.

## Why this matters

Prefill and decode place opposite demands on a GPU. Prefill is a single large, compute-bound pass whose latency appears once as time to first token (TTFT). Decode consists of many small passes whose latency is felt continuously as time per output token (TPOT). On a shared replica, admitting a long prompt stalls every active decode stream for the duration of that prefill, and single-pool defenses like chunked prefill or tighter batch caps only trade one latency metric against the other.

Disaggregating into separate prefill and decode pools eliminates that interference and lets each pool be sized for its own latency objective, at the cost of transferring KV caches between them. Because that transfer and queuing detour only pays off for long prompts, production systems apply disaggregation conditionally based on prompt length.
