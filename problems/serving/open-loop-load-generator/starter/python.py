import asyncio
import math


async def generate_load(
    send, rate: float, n: int, uniforms: list[float], max_in_flight: int | None = None
) -> list[dict]:
    """Send n requests as a Poisson process of `rate` requests per second.

    send: async callable; `await send(i)` performs request i and returns when its
        response is complete, or raises an OSError if the request failed.
    rate: mean arrivals per second, > 0.
    n: number of requests, >= 0.
    uniforms: at least n numbers in [0, 1); uniforms[i] gives the gap before request i.
    max_in_flight: None, or the largest number of requests that may be in flight at
        once (>= 1); a request that is due while that many are in flight waits.

    Returns a list of n dicts in request order, each with the keys "intended_s",
    "sent_s", "done_s", "latency_s", and "ok". Returns when every request has ended.

    Time is the running loop's clock: `asyncio.get_running_loop().time()` and
    `asyncio.sleep`.
    """
    raise NotImplementedError
