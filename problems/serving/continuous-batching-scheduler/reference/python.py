import asyncio
from collections import deque


async def run(engine, queue: asyncio.Queue, max_running: int) -> None:
    waiting = deque()  # requests that have arrived and hold no slot yet, oldest first
    running = []  # ids of admitted, unfinished sequences, in admission order
    while True:
        if not waiting and not running and queue.empty():
            # Idle: sleep until a request arrives. Every arrival of that instant is in
            # the queue by the time this coroutine resumes.
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        # Admission is per step, not per batch: any slot that is free now is filled.
        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        # One forward pass: new prompts are prefilled, everything older decodes.
        result = await engine.step(prefill=prefill, decode=running)

        # A sequence admitted in this step decodes from the next one on, and a sequence
        # that just finished gives its slot back before the next step is planned.
        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
