# BUG: the decode replica is chosen by outstanding tokens instead of requests in the house, so a replica full of nearly finished long outputs is preferred over one with a free slot.
class Coordinator:
    def __init__(self, cluster, prefill_ids, decode_ids, local_threshold):
        self.cluster = cluster
        self.prefill_pool = [cluster.replica(i) for i in prefill_ids]
        self.decode_pool = [cluster.replica(i) for i in decode_ids]
        self.local_threshold = local_threshold
        # KV caches on their way to each decode replica: load it cannot see yet.
        self.inbound = dict.fromkeys(decode_ids, 0)

    def pick_decode(self):
        return min(
            self.decode_pool, key=lambda r: (r.outstanding_tokens + self.inbound[r.id], r.id)
        )

    async def serve(self, request) -> None:
        if request.prompt_tokens < self.local_threshold:
            # A short prompt costs a decode replica less than moving it would.
            await self.pick_decode().submit(request)
            return

        prefill = min(self.prefill_pool, key=lambda r: (r.outstanding_tokens, r.id))
        await prefill.submit(request, phase="prefill")  # returns with the first token

        # Only now is it known when the decode replica will be needed: choose it by
        # the load at this instant, and count the transfer as load until it lands.
        decode = self.pick_decode()
        self.inbound[decode.id] += 1
        await self.cluster.transfer(request, prefill, decode)
        self.inbound[decode.id] -= 1
        await decode.submit(request, phase="decode")
