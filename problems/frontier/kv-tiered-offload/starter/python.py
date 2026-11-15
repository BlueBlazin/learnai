class TieredKV:
    """An index over KV blocks that live in three tiers: GPU, host memory, and disk.

    `store` holds the blocks and does the moving; this class decides.
        store.capacity    {"gpu": n, "host": n, "disk": n}, in blocks
        store.costs       {"compute_s", "host_base_s", "host_block_s",
                           "disk_base_s", "disk_block_s"}, in seconds
        await store.compute(key, parent, tokens)   compute one block on the GPU
        await store.load(keys)                     one transfer from host or disk to the GPU
        store.demote(key, tier)                    move a block down to "host" or "disk"
        store.drop(key)                            delete a block
    """

    def __init__(self, store, block_size: int):
        self.store = store
        self.block_size = block_size

    async def serve(self, tokens: list[int]) -> dict:
        """Bring every full block of `tokens` onto the GPU: reuse, load, or compute.

        Returns {"gpu": g, "host": h, "disk": d, "computed": c}: how many blocks were
        reused in place, loaded from the host tier, loaded from disk, and computed.
        """
        raise NotImplementedError

    def discard(self, tokens: list[int]) -> bool:
        """Drop the last full block of `tokens` from whatever tier holds it.

        Returns True if a block was dropped, False if it was not stored (or `tokens`
        has no full block).
        """
        raise NotImplementedError
