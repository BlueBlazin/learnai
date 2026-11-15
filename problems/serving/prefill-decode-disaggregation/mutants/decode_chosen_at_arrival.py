# BUG: the decode replica is chosen when the request arrives and reserved while the prompt is still being prefilled, by load that is out of date when the transfer starts; meanwhile the reservation keeps other requests away from an idle replica.
class Coordinator:
    def __init__(self, cluster, prefill_ids, decode_ids, local_threshold):
        self.cluster = cluster
        self.prefill_pool = [cluster.replica(i) for i in prefill_ids]
        self.decode_pool = [cluster.replica(i) for i in decode_ids]
        self.local_threshold = local_threshold
        # KV caches on their way to each decode replica: load it cannot see yet.
        self.inbound = dict.fromkeys(decode_ids, 0)

    def pick_decode(self):
        return min(self.decode_pool, key=lambda r: (r.queue_len + self.inbound[r.id], r.id))

    async def serve(self, request) -> None:
        if request.prompt_tokens < self.local_threshold:
            # A short prompt costs a decode replica less than moving it would.
            await self.pick_decode().submit(request)
            return

        prefill = min(self.prefill_pool, key=lambda r: (r.outstanding_tokens, r.id))
        decode = self.pick_decode()
        self.inbound[decode.id] += 1
        await prefill.submit(request, phase="prefill")  # returns with the first token
        await self.cluster.transfer(request, prefill, decode)
        self.inbound[decode.id] -= 1
        await decode.submit(request, phase="decode")
