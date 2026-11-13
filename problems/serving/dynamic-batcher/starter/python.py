from collections import deque


class DynamicBatcher:
    """A dynamic batching policy, driven by a simulator. It never sleeps or blocks.

    The simulator calls one of the three `on_*` handlers for every event, in order of
    virtual time. Each handler returns the batch to start on the model right now (a
    list of request ids, oldest first), or None to start nothing. After every handler
    call the simulator asks `next_timer_ms()` when to call `on_timer`.

    The policy: whenever the model is idle and requests are waiting, dispatch as soon
    as `max_batch_size` requests are waiting or the oldest waiting request has waited
    `max_wait_ms`, whichever happens first. A batch is the oldest waiting requests, at
    most `max_batch_size` of them.
    """

    def __init__(self, max_batch_size: int, max_wait_ms: int):
        """max_batch_size >= 1: the most requests one batch may hold.
        max_wait_ms >= 0: how long the oldest waiting request may wait at an idle model.
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    def on_arrival(self, now_ms: int, request_id) -> list | None:
        """A request arrived at virtual time `now_ms` (an int, in milliseconds).

        request_id: an opaque hashable value, unique per request.
        Returns the batch to dispatch now, or None.
        """
        raise NotImplementedError

    def on_batch_done(self, now_ms: int) -> list | None:
        """The model finished the batch it was running; it is idle again.

        Returns the next batch to dispatch now, or None.
        """
        raise NotImplementedError

    def on_timer(self, now_ms: int) -> list | None:
        """The time most recently returned by `next_timer_ms()` has been reached.

        Returns the batch to dispatch now, or None.
        """
        raise NotImplementedError

    def next_timer_ms(self) -> int | None:
        """The virtual time (int, ms) at which the simulator should call `on_timer`.

        Called after every handler call; only the most recent answer counts. Return
        None when no wake-up is needed. A time that is not in the future is ignored.
        """
        raise NotImplementedError
