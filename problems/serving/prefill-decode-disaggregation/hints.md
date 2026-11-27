A disaggregated request performs three awaits in sequence: prefill, KV transfer, and decode. Because `serve` runs as one coroutine per request, you can write those three steps straight down in order. Focus on what decision happens before each `await` and what in-flight state the coordinator must track that the replicas cannot report themselves.

---

Pick the prefill replica by `outstanding_tokens` when `serve` starts. Do not pick the decode replica yet, because queue lengths at arrival will be hundreds of milliseconds stale by the time prefill finishes. Instead, choose the decode replica right after prefill returns. Because a request in transfer is not in any replica's queue, `queue_len` cannot see it. Maintain a per-decode-replica counter of transfers in flight, increment it when you pick the replica, decrement it when `cluster.transfer` returns, and add it to `queue_len` whenever you compare decode replicas (including for short local requests).

---

```python
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

Here `pick_decode` returns `min(self.decode_pool, key=lambda r: (r.queue_len + self.inbound[r.id], r.id))`.
