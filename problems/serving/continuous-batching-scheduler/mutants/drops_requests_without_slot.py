# BUG: there is no waiting list: a request taken from the queue while every slot is taken is discarded, so arrivals during a full step are lost.
import asyncio


async def run(engine, queue: asyncio.Queue, max_running: int) -> None:
    running = []
    while True:
        arrived = []
        if not running and queue.empty():
            arrived.append(await queue.get())
        while not queue.empty():
            arrived.append(queue.get_nowait())

        prefill = []
        for request in arrived:
            if len(running) + len(prefill) < max_running:
                engine.admit(request)
                prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)

        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
