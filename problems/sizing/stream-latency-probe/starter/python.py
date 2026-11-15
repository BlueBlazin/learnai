import asyncio


def now() -> float:
    """The current time in seconds, read from the running event loop's clock.

    Under the judge this clock is virtual. Use it for every timestamp; never the real
    clock.
    """
    return asyncio.get_running_loop().time()


async def probe(open_stream, sink) -> dict:
    """Send one request, consume its stream, and return its latency record.

    open_stream: an async callable without arguments. Awaiting it sends the request
        and returns an async iterator of chunks; a chunk is a list of token ids, and
        may be empty. It may raise OSError, and so may the iteration.
    sink: an async callable. Every chunk is passed to it, in order, before the next
        chunk is requested.
    Returns a dict with the keys "status" ("ok", "empty", or "error"), "t_start",
    "token_times", "ttft", "itl", "tpot", and "e2e".
    """
    raise NotImplementedError


def summarize(results: list[dict], q: float) -> dict:
    """The q-th percentile (linear interpolation) of each metric over the "ok" records.

    Returns a dict with the keys "requests", "ok", "ttft", "tpot", "itl", and "e2e".
    """
    raise NotImplementedError


async def run_closed_loop(open_request, sink, concurrency: int, n: int) -> list[dict]:
    """Probe requests 0..n-1 with at most `concurrency` in flight at any time.

    open_request: an async callable of one argument; open_request(i) sends request i
        and returns its stream, like open_stream in probe.
    Returns the n records in request order.
    """
    raise NotImplementedError
