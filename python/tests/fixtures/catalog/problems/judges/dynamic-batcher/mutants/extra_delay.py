# BUG: batches correctly but holds every batch for a fixed extra delay before dispatch.
import asyncio


class Batcher:
    def __init__(self, model, max_batch_size, max_wait_s):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s
        self.pending = []
        self.timer = None
        self.model_lock = asyncio.Lock()

    async def submit(self, item):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending.append((item, future))
        if len(self.pending) >= self.max_batch_size:
            self._dispatch()
        elif self.timer is None:
            self.timer = loop.call_later(self.max_wait_s, self._dispatch)
        return await future

    def _dispatch(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        batch, self.pending = self.pending, []
        if batch:
            asyncio.ensure_future(self._run(batch))

    async def _run(self, batch):
        await asyncio.sleep(0.2)
        async with self.model_lock:
            results = await self.model([item for item, _ in batch])
        for (_, future), result in zip(batch, results, strict=True):
            future.set_result(result)
