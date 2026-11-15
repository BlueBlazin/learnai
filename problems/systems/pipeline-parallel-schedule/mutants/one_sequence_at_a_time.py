# BUG: a sequence occupies the whole pipeline until its last token is done; the stages never work on different sequences at once, so all but one of them are always idle.
import asyncio


async def run_pipeline(pipe, requests: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    n_stages = pipe.n_stages
    loop = asyncio.get_running_loop()
    outputs = {}

    for seq, arrival, n_tokens in requests:
        if loop.time() < arrival:
            await asyncio.sleep(arrival - loop.time())
        tokens = []
        for token in range(n_tokens):
            result = None
            for stage in range(n_stages):
                result = await pipe.run(stage, seq, token)
                if stage < n_stages - 1:
                    await pipe.transfer(stage, seq, token)
            tokens.append(result)
        outputs[seq] = tokens

    return outputs
