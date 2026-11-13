# BUG: the deadline is set only when a request arrives at an empty queue and is not recomputed after a dispatch; requests left over after a full batch inherit the deadline of a request that has already left and are flushed too early.
from collections import deque


class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: int):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.busy = False
        self.deadline = None

    def on_arrival(self, now_ms: int, request_id) -> list | None:
        if not self.queue:
            self.deadline = now_ms + self.max_wait_ms
        self.queue.append(request_id)
        return self._dispatch_if_due(now_ms)

    def on_batch_done(self, now_ms: int) -> list | None:
        self.busy = False
        return self._dispatch_if_due(now_ms)

    def on_timer(self, now_ms: int) -> list | None:
        return self._dispatch_if_due(now_ms)

    def next_timer_ms(self) -> int | None:
        if self.busy or not self.queue:
            return None
        return self.deadline

    def _dispatch_if_due(self, now_ms: int) -> list | None:
        if self.busy or not self.queue:
            return None
        full = len(self.queue) >= self.max_batch_size
        expired = now_ms >= self.deadline
        if not (full or expired):
            return None
        size = min(len(self.queue), self.max_batch_size)
        batch = [self.queue.popleft() for _ in range(size)]
        self.busy = True
        return batch
