# BUG: the prefill replica is chosen by its number of requests instead of its outstanding tokens, so a replica holding one 4000-token prompt looks less busy than one holding two short ones.
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

        prefill = min(self.prefill_pool, key=lambda r: (r.queue_len, r.id))
        await prefill.submit(request, phase="prefill")  # returns with the first token

        # Only now is it known when the decode replica will be needed: choose it by
        # the load at this instant, and count the transfer as load until it lands.
        decode = self.pick_decode()
        self.inbound[decode.id] += 1
        await self.cluster.transfer(request, prefill, decode)
        self.inbound[decode.id] -= 1
        await decode.submit(request, phase="decode")
