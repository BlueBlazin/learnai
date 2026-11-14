# BUG: reads every token and its log-probability with .item() inside the loop, so each step is evaluated implicitly (twice) and mx.eval is never called.
import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    logits, cache = model(mx.array(prompt))
    logprob = 0.0
    tokens = []
    for step in range(n):
        last = logits[-1]
        logprobs = last - mx.logsumexp(last)
        token = mx.argmax(logprobs).item()
        logprob += logprobs[token].item()
        tokens.append(token)
        if step + 1 < n:
            logits, cache = model(mx.array([token]), cache)
    return [mx.array(token) for token in tokens], mx.array(logprob)
