import asyncio


async def run_pipeline(pipe, requests: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    """Generate every token of every request on a pipeline of stages, interleaved.

    pipe: the hardware.
        pipe.n_stages; pipe.service_ms[i]: time stage i needs for one token;
        pipe.transfer_ms[i]: time to move an activation from stage i to stage i + 1.
        await pipe.run(stage, seq, token): stage computes that token; returns the
            token id when stage is the last one, else None.
        await pipe.transfer(stage, seq, token): moves the stage's output to stage + 1.
    requests: (seq_id, arrival_ms, n_tokens) tuples, sorted by arrival_ms.

    One unit of the event loop's clock is one millisecond, and it reads 0 now.
    Returns {seq_id: [token ids, in order]}, once every token has been generated.
    """
    raise NotImplementedError
