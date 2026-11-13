# BUG: only dispatches full batches; a partial batch waits forever.
import asyncio


class Batcher:
    def __init__(self, model, max_batch_size, max_wait_s):
        self.model = model
        self.max_batch_size = max_batch_size
        self.pending = []
        self.model_lock = asyncio.Lock()

    async def submit(self, item):
        future = asyncio.get_running_loop().create_future()
        self.pending.append((item, future))
        if len(self.pending) >= self.max_batch_size:
            batch, self.pending = self.pending, []
            asyncio.ensure_future(self._run(batch))
        return await future

    async def _run(self, batch):
        async with self.model_lock:
            results = await self.model([item for item, _ in batch])
        for (_, future), result in zip(batch, results, strict=True):
            future.set_result(result)
