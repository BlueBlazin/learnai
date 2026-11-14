# BUG: static batching: waiting requests are admitted only when nothing is running, so a free slot stays empty until the whole previous batch has finished.
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
        while not running and waiting and len(prefill) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
