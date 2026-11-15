# BUG: a global lock lets only one token travel through the pipeline at a time; sequences take turns token by token, but while one stage works the others wait.
import asyncio


async def run_pipeline(pipe, requests: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    n_stages = pipe.n_stages
    pipeline = asyncio.Lock()
    outputs = {}

    async def sequence(seq: str, arrival: int, n_tokens: int) -> None:
        await asyncio.sleep(arrival)
        tokens = []
        for token in range(n_tokens):
            result = None
            async with pipeline:
                for stage in range(n_stages):
                    result = await pipe.run(stage, seq, token)
                    if stage < n_stages - 1:
                        await pipe.transfer(stage, seq, token)
            tokens.append(result)
        outputs[seq] = tokens

    await asyncio.gather(*(sequence(*request) for request in requests))
    return outputs
