# BUG: the pools are ignored: every request runs from start to end on the least loaded replica of the whole cluster, which is the colocated baseline.
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
        everyone = [*self.prefill_pool, *self.decode_pool]
        await min(everyone, key=lambda r: (r.queue_len, r.id)).submit(request)
