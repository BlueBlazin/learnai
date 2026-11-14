import asyncio
from collections import deque


def now() -> float:
    """The current time in seconds, read from the running event loop's clock.

    Under the judge this clock is virtual. Use it, and asyncio's timers
    (`loop.call_later`, `asyncio.sleep`, `asyncio.wait_for`), for everything that
    involves time; never the real clock.
    """
    return asyncio.get_running_loop().time()


class TokenBucket:
    """A rate limiter that charges in LLM tokens.

    The bucket holds a balance of at most `burst` tokens. It starts full and refills
    continuously at `rate_per_s` tokens per second. Every method is called from inside
    the running event loop.
    """

    def __init__(self, rate_per_s: float, burst: int):
        """rate_per_s: refill rate in tokens per second, > 0.
        burst: the capacity of the bucket in tokens, an int >= 1.
        """
        raise NotImplementedError

    def try_acquire(self, n: int) -> bool:
        """Take n tokens now if that is possible, without ever waiting.

        Returns True and deducts n if no acquire() call is waiting and the balance is
        at least n; otherwise returns False and changes nothing.
        Raises ValueError if n > burst.
        """
        raise NotImplementedError

    async def acquire(self, n: int) -> None:
        """Take n tokens, waiting for them if necessary; returns when they are taken.

        Waiting calls are served strictly first in, first out: a call returns at the
        earliest time at which every earlier call has been served and the balance is
        at least n. If nobody is waiting and the balance suffices, it returns at once.
        Raises ValueError if n > burst, immediately and without waiting.
        """
        raise NotImplementedError

    def refund(self, n: int) -> None:
        """Give back n unused tokens (n >= 0).

        The balance grows by n, capped at burst. Waiting acquire() calls that can now
        be served, in order, return at this same instant.
        """
        raise NotImplementedError
