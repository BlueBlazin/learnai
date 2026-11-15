# BUG: the transfer is started only once the next stage has been claimed, so the activation waits on the sending device and the receiving stage sits idle for the whole transfer.
import asyncio


async def run_pipeline(pipe, requests: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    n_stages = pipe.n_stages
    locks = [asyncio.Lock() for _ in range(n_stages)]
    outputs = {}

    async def sequence(seq: str, arrival: int, n_tokens: int) -> None:
        await asyncio.sleep(arrival)
        tokens = []
        for token in range(n_tokens):
            result = None
            for stage in range(n_stages):
                async with locks[stage]:
                    if stage > 0:
                        await pipe.transfer(stage - 1, seq, token)
                    result = await pipe.run(stage, seq, token)
            tokens.append(result)
        outputs[seq] = tokens

    await asyncio.gather(*(sequence(*request) for request in requests))
    return outputs
