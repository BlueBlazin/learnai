# BUG: latency is measured from the moment the request was actually sent, so the time it waited for a free slot is left out (coordinated omission).
import asyncio
import math


async def generate_load(
    send, rate: float, n: int, uniforms: list[float], max_in_flight: int | None = None
) -> list[dict]:
    loop = asyncio.get_running_loop()
    slots = None if max_in_flight is None else asyncio.Semaphore(max_in_flight)
    records = [None] * n

    async def one(i: int, intended: float) -> None:
        if slots is not None:
            await slots.acquire()  # first come, first served; may make the send late
        sent = loop.time()
        try:
            await send(i)
            ok = True
        except OSError:
            ok = False  # a failed request is a data point, not the end of the test
        finally:
            if slots is not None:
                slots.release()
        done = loop.time()
        # Latency counts from when the request was due, not from when it got out:
        # the time it spent waiting for a slot is time its user waited.
        records[i] = {
            "intended_s": intended,
            "sent_s": sent,
            "done_s": done,
            "latency_s": done - sent,
            "ok": ok,
        }

    tasks = []
    intended = loop.time()
    for i in range(n):
        # A Poisson process: independent exponential gaps with mean 1 / rate. The
        # schedule depends on the uniforms alone, never on a response.
        intended += -math.log(1.0 - uniforms[i]) / rate
        delay = intended - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(loop.create_task(one(i, intended)))  # do not wait for it
    if tasks:
        await asyncio.wait(tasks)
    return records
