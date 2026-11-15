# BUG: nothing is ever shed: a request that cannot get its first token in time is admitted anyway and served late, taking a slot from requests that could still be on time.
import asyncio
import math


def deadline_of(request) -> float:
    """The absolute time by which the first token must appear; no deadline is +inf."""
    return math.inf if request.deadline_s is None else request.deadline_s


def plan_step(waiting: list, now_ms: float, running: int, free_slots: int, step_ms):
    # Earliest deadline first. `sorted` is stable, so equal deadlines keep the order
    # of `waiting`, which is arrival order.
    admit, shed = [], []
    for request in sorted(waiting, key=deadline_of):
        if len(admit) < free_slots:
            admit.append(request)  # served whether or not it can still be on time
    for request in (*admit, *shed):
        waiting.remove(request)
    return admit, shed


async def run(engine, queue: asyncio.Queue, max_running: int, on_shed) -> None:
    loop = asyncio.get_running_loop()
    waiting = []  # requests without a slot, in arrival order
    running = []  # ids of admitted, unfinished sequences, in admission order
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        admit, shed = plan_step(
            waiting, loop.time(), len(running), max_running - len(running), engine.cost.step_s
        )
        for request in shed:
            on_shed(request)  # told now, not after it has waited in vain
        if not admit and not running:
            continue  # everything that was waiting has been shed: nothing to run

        prefill = []
        for request in admit:
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
