class Scheduler:
    def __init__(self, max_batch: int):
        self.max_batch = max_batch

    def submit(self, request_id: int, tokens_needed: int) -> None:
        """Called by the simulator when a request arrives."""
        raise NotImplementedError

    def next_batch(self) -> list[int]:
        """Called before each step. Return the ids to run this step."""
        raise NotImplementedError

    def on_step_done(self, finished: list[int]) -> None:
        """Called after each step with the ids that produced their last token."""
        raise NotImplementedError
