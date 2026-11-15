import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """How call_with_retry retries. Given; do not change it."""

    max_attempts: int = 4  # sends in total, the first one included; >= 1
    base_s: float = 0.5  # backoff ceiling before the first retry
    cap_s: float = 8.0  # the ceiling never exceeds this
    deadline_s: float = 30.0  # no send starts this long after the call began, or later
    idempotent: bool = True  # may the request be resent when its outcome is unknown?


async def call_with_retry(send, policy: RetryPolicy, uniforms: list[float], budget=None):
    """Send a request, retrying overload responses with backoff and full jitter.

    send: an async callable without arguments. It returns a response with `.status`
        (int) and `.headers` (dict of str to str), or raises ConnectionError.
    uniforms: uniform numbers in [0, 1); the k-th backoff delay this call computes
        uses uniforms[k]. A delay taken from Retry-After uses none.
    budget: None, or a shared retry budget with success(), failure(), and
        allows_retry() -> bool.
    Returns the last response, or raises the last ConnectionError.

    Time is the running event loop's clock (asyncio.get_running_loop().time(),
    asyncio.sleep); under the judge it is virtual.
    """
    raise NotImplementedError
