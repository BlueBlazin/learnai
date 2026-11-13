import asyncio


class Batcher:
    def __init__(self, model, max_batch_size, max_wait_s):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_s

    async def submit(self, item):
        """Wait for the model's result for this item."""
        raise NotImplementedError
