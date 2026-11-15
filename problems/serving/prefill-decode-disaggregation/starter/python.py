class Coordinator:
    """Serves requests on a cluster split into a prefill pool and a decode pool."""

    def __init__(
        self, cluster, prefill_ids: list[int], decode_ids: list[int], local_threshold: int
    ):
        """cluster: a SimCluster. prefill_ids, decode_ids: the replica ids of the two
        pools. local_threshold: prompts shorter than this many tokens are served
        entirely on a decode replica."""
        self.cluster = cluster
        self.prefill_ids = list(prefill_ids)
        self.decode_ids = list(decode_ids)
        self.local_threshold = local_threshold

    async def serve(self, request) -> None:
        """Serve one request; return when it has finished.

        This starter ignores the pools: every request runs from start to end on
        replica 0.
        """
        await self.cluster.replica(0).submit(request)
