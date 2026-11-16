import asyncio
from collections import deque


async def run(engine, queue: asyncio.Queue, max_running: int) -> None:
    """The scheduling loop of a continuous-batching server. It never returns.

    engine: a `learnai.kit.SimEngine`. The calls you need:
        engine.admit(request)        make the request a live sequence; takes no time
        result = await engine.step(prefill=[(id, n_tokens), ...], decode=[id, ...])
                                     one forward pass; returns when it ends
        result.finished              ids that produced their last token in that step
    queue: an `asyncio.Queue` of `SimRequest`s (`id`, `prompt_tokens`, ...). The test
        puts each request into it at its arrival time; arrivals of one instant are put
        in arrival order.
    max_running: the number of slots: at most this many sequences are live at once.

    Before every step: move everything in `queue` to the end of the waiting list, give
    each free slot to the oldest waiting request (admit it and prefill its whole
    prompt in this step), and decode every sequence admitted in an earlier step, in
    admission order. A finished sequence frees its slot for the very next step. With
    nothing waiting and nothing running, block on `queue.get()`; never run an empty
    step and never sleep.
    """
    raise NotImplementedError
