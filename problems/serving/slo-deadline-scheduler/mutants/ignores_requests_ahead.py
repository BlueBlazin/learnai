# BUG: the first-token estimate of a request counts only its own prompt, not the prompts already admitted to the same step ahead of it, so the step is longer than estimated and requests are served late.
import asyncio
import math


def deadline_of(request) -> float:
    """The absolute time by which the first token must appear; no deadline is +inf."""
    return math.inf if request.deadline_s is None else request.deadline_s


def plan_step(waiting: list, now_ms: float, running: int, free_slots: int, step_ms):
    # Earliest deadline first. `sorted` is stable, so equal deadlines keep the order
    # of `waiting`, which is arrival order.
    admit, shed = [], []
    prefill_tokens = 0
    for request in sorted(waiting, key=deadline_of):
        # When the first token would appear if this request joined the step behind
        # everything already admitted to it. The step also decodes `running` sequences.
        first_token = now_ms + step_ms(request.prompt_tokens, running)
        if first_token > deadline_of(request):
            # It cannot be on time in this step, and any later step ends later still.
            shed.append(request)
        elif len(admit) < free_slots and (not admit or first_token <= deadline_of(admit[0])):
            # It fits, and the most urgent request of the step (the first admitted)
            # still makes its deadline with the longer prefill.
            admit.append(request)
            prefill_tokens += request.prompt_tokens
        # Otherwise it stays waiting: it can still be on time in a later step.
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
