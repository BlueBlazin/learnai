# BUG: a free slot goes to the newest waiting request instead of the oldest (the waiting list is used as a stack), so admission is not first come, first served.
import asyncio
from collections import deque


async def run(engine, queue: asyncio.Queue, max_running: int) -> None:
    waiting = deque()
    running = []
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = waiting.pop()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
