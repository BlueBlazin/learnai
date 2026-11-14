# BUG: the slots of sequences that finished in the last step are returned to the free count after this step's admissions, so each freed slot stays empty for one extra step.
import asyncio
from collections import deque


async def run(engine, queue: asyncio.Queue, max_running: int) -> None:
    waiting = deque()
    running = []
    free = max_running
    just_finished = 0
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        prefill = []
        while waiting and free > 0:
            request = waiting.popleft()
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))
            free -= 1
        free += just_finished
        just_finished = 0
        if not prefill and not running:
            continue

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        just_finished = len(finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
