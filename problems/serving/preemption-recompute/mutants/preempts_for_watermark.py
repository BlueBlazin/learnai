# BUG: running sequences are preempted to keep the watermark free, although the watermark only limits admission: sequences are evicted while the pool still has room.
import asyncio
from collections import deque


async def run(engine, queue: asyncio.Queue, max_running: int, watermark_blocks: int) -> None:
    block = engine.block_tokens
    waiting = deque()  # requests without memory, next to run first
    running = []  # requests that hold KV blocks, in the order they were (re)admitted
    preempted = set()  # ids of waiting requests that are still live in the engine

    while True:
        if not waiting and not running and queue.empty():
            waiting.append(await queue.get())
        while not queue.empty():
            waiting.append(queue.get_nowait())

        # 1. Memory for the running sequences, oldest first. Each decodes one token,
        #    which needs a new block exactly when its last block is full.
        reserved = 0  # blocks this step has promised but the engine has not taken yet
        evicted = False
        decode = []
        for request in running[:]:  # a copy: victims leave `running`
            if request not in running:
                break  # it was preempted for an older sequence; so was everything after it
            if engine.kv_tokens(request.id) % block == 0:
                while engine.free_blocks - reserved < 1 + watermark_blocks and running:
                    victim = running.pop()  # the most recently admitted
                    engine.preempt(victim.id)  # frees its blocks, keeps its tokens
                    preempted.add(victim.id)
                    waiting.appendleft(victim)  # first in line to come back
                    evicted = True
                    if victim is request:
                        break
                if request not in running:
                    break
                reserved += 1
            decode.append(request.id)

        # 2. Admission, first come first served, unless this step had to preempt.
        prefill = []
        while not evicted and waiting and len(running) < max_running:
            request = waiting[0]
            if request.id in preempted:
                tokens = engine.remaining_prefill(request.id)  # prompt + what it produced
            else:
                tokens = request.prompt_tokens
            need = -(-tokens // block)
            # Leave headroom for the sequences that are already in this step to grow.
            headroom = watermark_blocks if decode or prefill else 0
            if engine.free_blocks - reserved - need < headroom:
                break
            waiting.popleft()
            if request.id in preempted:
                preempted.discard(request.id)
            else:
                engine.admit(request)
            reserved += need
            prefill.append((request.id, tokens))
            running.append(request)

        result = await engine.step(prefill=prefill, decode=decode)
        finished = set(result.finished)
        running = [request for request in running if request.id not in finished]
