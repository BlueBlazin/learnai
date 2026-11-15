import asyncio


async def run(engine, queue: asyncio.Queue, max_running: int, watermark_blocks: int) -> None:
    """Continuous batching on an engine whose KV cache is a finite pool of blocks.

    engine: a SimEngine with `kv_blocks` blocks of `engine.block_tokens` tokens.
    queue: SimRequest objects arrive here.
    max_running: the most sequences that may hold KV blocks at once.
    watermark_blocks: free blocks to keep when admitting next to other sequences.

    This starter is the loop of `continuous-batching-scheduler`. It ignores memory:
    as soon as the pool is too small for what it admits, the engine refuses the step.
    """
    waiting = []
    running = []
    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        prefill = []
        while waiting and len(running) + len(prefill) < max_running:
            request = waiting.pop(0)
            engine.admit(request)
            prefill.append((request.id, request.prompt_tokens))

        result = await engine.step(prefill=prefill, decode=running)
        finished = set(result.finished)
        running = [i for i in [*running, *(i for i, _ in prefill)] if i not in finished]
