# BUG: only full batches are dispatched: the wait deadline is never checked and no timer is requested, so a partial batch waits until enough requests arrive (or forever).
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
        return self._dispatch_if_due(now_ms)

    def on_timer(self, now_ms: int) -> list | None:
        return self._dispatch_if_due(now_ms)

    def next_timer_ms(self) -> int | None:
        return None

    def _deadline(self) -> int:
        return self.queue[0][1] + self.max_wait_ms

    def _dispatch_if_due(self, now_ms: int) -> list | None:
        if self.busy or not self.queue:
            return None
        full = len(self.queue) >= self.max_batch_size
        if not full:
            return None
        size = min(len(self.queue), self.max_batch_size)
        batch = [self.queue.popleft()[0] for _ in range(size)]
        self.busy = True
        return batch
