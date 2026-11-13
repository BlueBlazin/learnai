from collections import deque


class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: int):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()  # (request_id, arrival_ms) of waiting requests, oldest first
        self.busy = False  # True from the moment a batch is returned until on_batch_done

    def on_arrival(self, now_ms: int, request_id) -> list | None:
        self.queue.append((request_id, now_ms))
        return self._dispatch_if_due(now_ms)

    def on_batch_done(self, now_ms: int) -> list | None:
        self.busy = False
        return self._dispatch_if_due(now_ms)

    def on_timer(self, now_ms: int) -> list | None:
        return self._dispatch_if_due(now_ms)

    def next_timer_ms(self) -> int | None:
        # A wake-up is needed only while requests wait at an idle model. Then it is the
        # deadline of the oldest request, which is recomputed from the queue on every
        # call, so there is no timer state that could go stale.
        if self.busy or not self.queue:
            return None
        return self._deadline()

    def _deadline(self) -> int:
        return self.queue[0][1] + self.max_wait_ms

    def _dispatch_if_due(self, now_ms: int) -> list | None:
        # The whole policy is this one rule, evaluated after every event.
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
