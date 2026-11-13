class Scheduler:
    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.waiting: list[int] = []
        self.running: list[int] = []

    def submit(self, request_id: int, tokens_needed: int) -> None:
        self.waiting.append(request_id)

    def next_batch(self) -> list[int]:
        while self.waiting and len(self.running) < self.max_batch:
            self.running.append(self.waiting.pop(0))
        return list(self.running)

    def on_step_done(self, finished: list[int]) -> None:
        for request_id in finished:
            self.running.remove(request_id)
