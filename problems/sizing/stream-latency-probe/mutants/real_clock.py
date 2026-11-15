# BUG: timestamps are read from the real clock (time.monotonic) instead of the event loop's clock, which the statement forbids; under a simulated clock every latency comes out as almost zero.
import asyncio
import math
import time
from functools import partial
from itertools import pairwise


def now() -> float:
    return time.monotonic()


async def probe(open_stream, sink) -> dict:
    t_start = now()  # before the first await: connecting and queueing are part of TTFT
    token_times = []
    status = "ok"
    try:
        stream = await open_stream()
        async for chunk in stream:
            arrived = now()  # on arrival, before the consumer spends time on the chunk
            token_times.extend([arrived] * len(chunk))  # an empty chunk adds nothing
            await sink(chunk)
    except OSError:
        status = "error"
    else:
        if not token_times:
            status = "empty"

    result = {
        "status": status,
        "t_start": t_start,
        "token_times": token_times,
        "ttft": None,
        "itl": [],
        "tpot": None,
        "e2e": None,
    }
    if status == "ok":  # only a complete stream with tokens has latencies
        n = len(token_times)
        first, last = token_times[0], token_times[-1]
        result["ttft"] = first - t_start
        result["itl"] = [later - earlier for earlier, later in pairwise(token_times)]
        result["tpot"] = (last - first) / (n - 1) if n > 1 else None
        result["e2e"] = last - t_start  # the last token, not the end of the stream
    return result


def percentile_linear(values: list[float], q: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def summarize(results: list[dict], q: float) -> dict:
    ok = [r for r in results if r["status"] == "ok"]

    def pct(values):
        return percentile_linear(values, q) if values else None

    return {
        "requests": len(results),
        "ok": len(ok),
        "ttft": pct([r["ttft"] for r in ok]),
        "tpot": pct([r["tpot"] for r in ok if r["tpot"] is not None]),
        "itl": pct([gap for r in ok for gap in r["itl"]]),  # every gap of every request
        "e2e": pct([r["e2e"] for r in ok]),
    }


async def run_closed_loop(open_request, sink, concurrency: int, n: int) -> list[dict]:
    results = [None] * n
    next_index = 0

    async def worker():
        nonlocal next_index
        while next_index < n:
            i = next_index
            next_index += 1
            # The next request is sent the moment this one is over, not before.
            results[i] = await probe(partial(open_request, i), sink)

    await asyncio.gather(*(worker() for _ in range(min(concurrency, n))))
    return results
