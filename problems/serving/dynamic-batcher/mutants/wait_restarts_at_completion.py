# BUG: when a batch finishes, the requests still waiting have their wait restarted from the completion time, so a request that queued behind a running batch waits up to max_wait_ms again at an idle model.
from collections import deque


class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: int):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.busy = False

    def on_arrival(self, now_ms: int, request_id) -> list | None:
        self.queue.append((request_id, now_ms))
        return self._dispatch_if_due(now_ms)

    def on_batch_done(self, now_ms: int) -> list | None:
        self.busy = False
        if len(self.queue) < self.max_batch_size:
            self.queue = deque((request_id, now_ms) for request_id, _ in self.queue)
        return self._dispatch_if_due(now_ms)

    def on_timer(self, now_ms: int) -> list | None:
        return self._dispatch_if_due(now_ms)

    def next_timer_ms(self) -> int | None:
        if self.busy or not self.queue:
            return None
        return self._deadline()

    def _deadline(self) -> int:
        return self.queue[0][1] + self.max_wait_ms

    def _dispatch_if_due(self, now_ms: int) -> list | None:
        if self.busy or not self.queue:
            return None
        full = len(self.queue) >= self.max_batch_size
        expired = now_ms >= self._deadline()
        if not (full or expired):
            return None
        size = min(len(self.queue), self.max_batch_size)
        batch = [self.queue.popleft()[0] for _ in range(size)]
        self.busy = True
        return batch
