from collections import deque


class Scheduler:
    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.waiting: deque[int] = deque()
        self.running: dict[int, None] = {}

    def submit(self, request_id: int, tokens_needed: int) -> None:
        self.waiting.append(request_id)

    def next_batch(self) -> list[int]:
        while self.waiting and len(self.running) < self.max_batch:
            self.running[self.waiting.popleft()] = None
        return list(self.running)

    def on_step_done(self, finished: list[int]) -> None:
        for request_id in finished:
            del self.running[request_id]
