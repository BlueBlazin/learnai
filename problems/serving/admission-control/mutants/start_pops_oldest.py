# BUG: start removes the oldest queued request instead of the request it was given, so the accounting is wrong whenever the server starts requests out of arrival order.
from dataclasses import dataclass


@dataclass(frozen=True)
class Admitted:
    """The request was accepted: it is now in the queue."""


@dataclass(frozen=True)
class Rejected:
    """The request was turned away; retry_after_s estimates when it would fit."""

    retry_after_s: float | None


class Admission:
    def __init__(self, max_queued_requests: int, max_queued_tokens: int, drain_tokens_per_s: float):
        self.max_queued_requests = max_queued_requests
        self.max_queued_tokens = max_queued_tokens
        self.drain_tokens_per_s = drain_tokens_per_s
        self._queue = {}  # id -> prompt_tokens; a dict keeps arrival order
        self._queued_tokens = 0
        self._running = set()

    def admit(self, req) -> Admitted | Rejected:
        # Decide first, enqueue second: a rejected request must leave no trace.
        if self._fits(req.prompt_tokens, len(self._queue), self._queued_tokens):
            self._queue[req.id] = req.prompt_tokens
            self._queued_tokens += req.prompt_tokens
            return Admitted()
        return Rejected(self._retry_after_s(req.prompt_tokens))

    def start(self, req) -> None:
        # From here on the request occupies the engine, not the queue.
        self._queued_tokens -= self._queue.pop(next(iter(self._queue)))
        self._running.add(req.id)

    def finish(self, req) -> None:
        self._running.remove(req.id)

    @property
    def queued_requests(self) -> int:
        return len(self._queue)

    @property
    def queued_tokens(self) -> int:
        return self._queued_tokens

    @property
    def running_requests(self) -> int:
        return len(self._running)

    def _fits(self, prompt_tokens: int, queued_requests: int, queued_tokens: int) -> bool:
        return (
            queued_requests + 1 <= self.max_queued_requests
            and queued_tokens + prompt_tokens <= self.max_queued_tokens
        )

    def _retry_after_s(self, prompt_tokens: int) -> float | None:
        if prompt_tokens > self.max_queued_tokens:
            return None  # no amount of waiting makes room for this request
        requests, tokens, drained = len(self._queue), self._queued_tokens, 0
        # Let the queue drain from its head until the request would fit; an empty
        # queue always fits, so the loop returns.
        for waiting in self._queue.values():
            requests -= 1
            tokens -= waiting
            drained += waiting
            if self._fits(prompt_tokens, requests, tokens):
                break
        return drained / self.drain_tokens_per_s
