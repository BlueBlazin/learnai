# BUG: the effective priorities are stored and refreshed only when a new request joins the waiting list; a slot handed out later, with no arrival in between, is decided on stale levels.
import asyncio


def effective_priority(request, now_ms: float, aging_ms: float) -> int:
    return request.priority - int((now_ms - request.arrival_s) // aging_ms)


def pick_next(waiting: list, now_ms: float, aging_ms: float):
    best = min(waiting, key=lambda r: (effective_priority(r, now_ms, aging_ms), r.arrival_s))
    waiting.remove(best)
    return best


async def run(engine, queue: asyncio.Queue, max_running: int, aging_ms: float) -> None:
    loop = asyncio.get_running_loop()
    waiting = []
    level = {}  # request id -> effective priority at the last refresh
    running = []
    while True:
        arrived = []
        if not waiting and not running and queue.empty():
            arrived.append(await queue.get())
        while not queue.empty():
            arrived.append(queue.get_nowait())
        if arrived:
            waiting.extend(arrived)
            now_ms = loop.time()
            level = {r.id: effective_priority(r, now_ms, aging_ms) for r in waiting}

        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = min(waiting, key=lambda r: (level[r.id], r.arrival_s))
            waiting.remove(request)
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
