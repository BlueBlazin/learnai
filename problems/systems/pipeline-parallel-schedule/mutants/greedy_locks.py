# BUG: one coroutine per sequence and one asyncio.Lock per stage: a stage goes to whoever asks first, so when a returning token and a new request become ready at the same instant the order is the event loop's, not (ready time, request order).
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
                    result = await pipe.run(stage, seq, token)
                if stage < n_stages - 1:
                    await pipe.transfer(stage, seq, token)
            tokens.append(result)
        outputs[seq] = tokens

    await asyncio.gather(*(sequence(*request) for request in requests))
    return outputs
