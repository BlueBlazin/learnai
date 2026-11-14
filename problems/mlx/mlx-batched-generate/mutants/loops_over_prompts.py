# BUG: generates one prompt at a time; the tokens are right, but the model is called once per row per step and the GPU never sees a batch.
import mlx.core as mx


def batch_generate(
    model, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    out: list[list[int]] = []
    for prompt in prompts:
        logits, cache = model(mx.array([prompt]))
        tokens: list[int] = []
        while True:
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token, cache)
            tokens.append(token.item())
            if tokens[-1] == eos_id or len(tokens) == max_new_tokens:
                break
            logits, cache = model(token[:, None], cache)
        out.append(tokens)
    return out
