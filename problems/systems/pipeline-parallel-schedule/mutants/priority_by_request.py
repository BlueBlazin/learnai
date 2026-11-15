# BUG: a free stage takes the waiting job of the earliest request instead of the job that has been ready longest, so later requests wait while an early sequence is served again and again.
import asyncio
import heapq

READY, FREE = 0, 1


async def run_pipeline(pipe, requests: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    n_stages = pipe.n_stages
    service, transfer = pipe.service_ms, pipe.transfer_ms
    loop = asyncio.get_running_loop()
    outputs = {seq: [0] * n_tokens for seq, _, n_tokens in requests}

    calendar = [(arrival, READY, 0, index, 0) for index, (_, arrival, _) in enumerate(requests)]
    heapq.heapify(calendar)
    waiting = [[] for _ in range(n_stages)]
    free = [True] * n_stages
    jobs = []

    async def job(stage: int, index: int, token: int) -> None:
        seq = requests[index][0]
        result = await pipe.run(stage, seq, token)
        if stage < n_stages - 1:
            await pipe.transfer(stage, seq, token)
        else:
            outputs[seq][token] = result

    while calendar:
        now = calendar[0][0]
        await asyncio.sleep(now - loop.time())

        while calendar and calendar[0][0] == now:
            _, kind, stage, index, token = heapq.heappop(calendar)
            if kind == FREE:
                free[stage] = True
            else:
                heapq.heappush(waiting[stage], (0, index, token))

        for stage in range(n_stages):
            if not (free[stage] and waiting[stage]):
                continue
            _, index, token = heapq.heappop(waiting[stage])
            free[stage] = False
            jobs.append(asyncio.create_task(job(stage, index, token)))
            done = now + service[stage]
            heapq.heappush(calendar, (done, FREE, stage, -1, -1))
            if stage < n_stages - 1:
                arrives = done + transfer[stage]
                heapq.heappush(calendar, (arrives, READY, stage + 1, index, token))
            elif token + 1 < requests[index][2]:
                heapq.heappush(calendar, (done, READY, 0, index, token + 1))

    await asyncio.gather(*jobs)
    return outputs
