import asyncio


class Batcher:
    """Collects requests into batches for a model that processes one batch at a time.

    A batch is dispatched as soon as it holds max_batch_size items, or max_wait_s
    after its first item arrived, whichever comes first.
    """

    def __init__(self, model, max_batch_size, max_wait_s):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s
        self.pending = []  # (item, future) pairs of the batch being collected
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
        async with self.model_lock:
            results = await self.model([item for item, _ in batch])
        for (_, future), result in zip(batch, results, strict=True):
            future.set_result(result)
