import asyncio


def plan_step(waiting: list, now_ms: float, running: int, free_slots: int, step_ms):
    """Decide what the step planned at `now_ms` does with the waiting requests.

    waiting: requests without a slot, in arrival order. `request.deadline_s` is the
        absolute time in ms by which the first token must appear, or None.
    now_ms: the virtual time at which the step would start.
    running: how many sequences the step decodes.
    free_slots: how many requests may be admitted, >= 0.
    step_ms: step_ms(prefill_tokens, decode_seqs) is the duration of a step in ms.

    Returns (admit, shed): the requests to admit, in prefill order, and the requests
    that can no longer get their first token by their deadline. Both are removed
    from `waiting`; the others stay, in their order.
    """
    raise NotImplementedError


async def run(engine, queue: asyncio.Queue, max_running: int, on_shed) -> None:
    """The continuous-batching loop with deadline-aware admission; it never returns.

    on_shed: call on_shed(request) for every request that plan_step sheds, at the
        instant the step is planned.
    """
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
            on_shed(request)
        if not admit and not running:
            continue

        prefill = []
        for request in admit:
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
