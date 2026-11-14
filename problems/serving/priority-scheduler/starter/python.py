import asyncio


def effective_priority(request, now_ms: float, aging_ms: float) -> int:
    """The priority of a waiting request at virtual time `now_ms`, after aging.

    request: a `SimRequest`. `request.priority` is an int, smaller is more urgent;
        `request.arrival_s` is its arrival time (in milliseconds in this problem).
    now_ms: the current virtual time in milliseconds, at least the arrival time.
    aging_ms: a positive number of milliseconds.

    Returns `request.priority` minus one for every full `aging_ms` the request has
    waited since it arrived. The result may be negative.
    """
    raise NotImplementedError


def pick_next(waiting: list, now_ms: float, aging_ms: float):
    """Remove and return the request that gets the next free slot.

    waiting: a non-empty list of `SimRequest`s in the order they left the queue.
        Remove the chosen request from it (in place).

    The choice: the smallest effective priority at `now_ms`; among equals the earliest
    `arrival_s`; among those the one that comes first in `waiting`.
    """
    raise NotImplementedError


async def run(engine, queue: asyncio.Queue, max_running: int, aging_ms: float) -> None:
    """The continuous-batching loop of `continuous-batching-scheduler`, complete.

    The only change is the admission line: a free slot goes to `pick_next(...)` and
    not to the oldest waiting request. You may restructure this loop (a heap, a class)
    as long as the steps it produces are the ones the statement defines.
    """
    loop = asyncio.get_running_loop()
    waiting = []  # requests without a slot, in the order they left the queue
    running = []  # ids of admitted, unfinished sequences, in admission order
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())  # idle: block until a request arrives
        while not queue.empty():
            waiting.append(queue.get_nowait())

        now_ms = loop.time()  # virtual time, in milliseconds
        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = pick_next(waiting, now_ms, aging_ms)
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
