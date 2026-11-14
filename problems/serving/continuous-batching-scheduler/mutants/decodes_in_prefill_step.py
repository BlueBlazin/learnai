# BUG: a newly admitted sequence is listed in `decode` in the same step that prefills it; it can only be decoded from the next step on.
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
        while waiting and len(running) < max_running:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
            running.append(request.id)

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in running if i not in finished]
