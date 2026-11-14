# BUG: waiting time is measured from the moment the scheduler takes the request out of the queue, not from its arrival, so a request that arrived during a step ages too slowly.
import asyncio


def effective_priority(request, now_ms: float, aging_ms: float) -> int:
    return request.priority - int((now_ms - request.arrival_s) // aging_ms)


async def run(engine, queue: asyncio.Queue, max_running: int, aging_ms: float) -> None:
    loop = asyncio.get_running_loop()
    waiting = []  # (request, time it was collected)
    running = []
    while True:
        if not waiting and not running and queue.empty():
            waiting.append((await queue.get(), loop.time()))
        while not queue.empty():
            waiting.append((queue.get_nowait(), loop.time()))

        now_ms = loop.time()
        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            entry = min(
                waiting,
                key=lambda e: (e[0].priority - int((now_ms - e[1]) // aging_ms), e[0].arrival_s),
            )
            waiting.remove(entry)
            engine.admit(entry[0])
            prefill.append((entry[0].id, entry[0].prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]


def pick_next(waiting: list, now_ms: float, aging_ms: float):
    best = min(waiting, key=lambda r: (effective_priority(r, now_ms, aging_ms), r.arrival_s))
    waiting.remove(best)
    return best
