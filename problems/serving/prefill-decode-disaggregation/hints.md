A request that is disaggregated goes through three awaits in a row: the prefill,
the transfer, the decode. `serve` is a coroutine per request, so it can simply be
written in that order. The questions are what is decided before each await, and
what the coordinator has to remember across them that the replicas cannot tell it.

---

Choose the prefill replica by tokens, at the start. Do not choose the decode
replica there: by the time the prefill ends, the queue lengths you looked at are
hundreds of milliseconds old. Choose it right after the prefill returns. A request
that is being transferred is in no replica's queue, so `queue_len` does not see it;
keep a counter per decode replica of transfers in flight, increment it when you
choose, decrement it when the transfer returns, and add it to `queue_len` whenever
you compare decode replicas (also for the short requests that go there directly).

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

with `pick_decode` returning `min(self.decode_pool, key=lambda r: (r.queue_len +
self.inbound[r.id], r.id))`.
