# BUG: equal effective priorities are ordered by request id instead of arrival time, so a later arrival with a smaller id overtakes an earlier one.
import asyncio


def effective_priority(request, now_ms: float, aging_ms: float) -> int:
    return request.priority - int((now_ms - request.arrival_s) // aging_ms)


def pick_next(waiting: list, now_ms: float, aging_ms: float):
    best = min(waiting, key=lambda r: (effective_priority(r, now_ms, aging_ms), str(r.id)))
    waiting.remove(best)
    return best


async def run(engine, queue: asyncio.Queue, max_running: int, aging_ms: float) -> None:
    loop = asyncio.get_running_loop()
    waiting = []  # requests without a slot, in the order they left the queue
    running = []  # ids of admitted, unfinished sequences, in admission order
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        # The priorities are evaluated now, when a slot is being handed out, because
        # they change while requests wait.
        now_ms = loop.time()
        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = pick_next(waiting, now_ms, aging_ms)
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
