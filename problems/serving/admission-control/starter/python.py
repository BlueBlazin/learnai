from dataclasses import dataclass


@dataclass(frozen=True)
class Admitted:
    """The request was accepted: it is now in the queue."""


@dataclass(frozen=True)
class Rejected:
    """The request was turned away.

    retry_after_s: the estimated number of seconds until the same request would fit,
        or None if it can never fit (its prompt alone exceeds max_queued_tokens).
    """

    retry_after_s: float | None


class Admission:
    """Bounds the queue in front of an inference server.

    A request is *queued* from the `admit` call that accepts it until `start`, and
    *running* from `start` until `finish`. Only queued requests count against the
    bounds. A request is any object with `id` (hashable, unique) and `prompt_tokens`
    (int >= 1), such as a `whetstone.kit.SimRequest`.
    """

    def __init__(self, max_queued_requests: int, max_queued_tokens: int, drain_tokens_per_s: float):
        """max_queued_requests: the most requests that may be queued, >= 1.
        max_queued_tokens: the most prompt tokens that may be queued in total, >= 1.
        drain_tokens_per_s: the estimated rate at which queued prompt tokens leave the
            queue, used only for `retry_after_s`.
        """
        raise NotImplementedError

    def admit(self, req) -> Admitted | Rejected:
        """Decide whether `req` may join the queue.

        Accept (and enqueue) if, with this request, the queue would hold at most
        max_queued_requests requests and at most max_queued_tokens prompt tokens.
        Otherwise return Rejected(retry_after_s) and leave the queue as it was.

        retry_after_s: None if req.prompt_tokens > max_queued_tokens. Otherwise find
        the smallest j such that the request would be accepted after the j oldest
        queued requests had left the queue; the estimate is the prompt tokens of
        those j requests divided by drain_tokens_per_s.
        """
        raise NotImplementedError

    def start(self, req) -> None:
        """The server begins to run `req`, a queued request (not necessarily the oldest).

        It leaves the queue and no longer counts against either bound.
        """
        raise NotImplementedError

    def finish(self, req) -> None:
        """The running request `req` has completed."""
        raise NotImplementedError

    @property
    def queued_requests(self) -> int:
        """How many requests are queued."""
        raise NotImplementedError

    @property
    def queued_tokens(self) -> int:
        """The total prompt tokens of the queued requests."""
        raise NotImplementedError

    @property
    def running_requests(self) -> int:
        """How many requests are running."""
        raise NotImplementedError
