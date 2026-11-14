# BUG: evaluates the prefill on its own before the loop, so n tokens take n + 1 evaluations.
import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    logits, cache = model(mx.array(prompt))
    mx.eval(logits, cache)
    logprob = mx.array(0.0)
    tokens = []
    for step in range(n):
        last = logits[-1]
        logprobs = last - mx.logsumexp(last)
        token = mx.argmax(logprobs)
        logprob = logprob + logprobs[token]
        mx.eval(token, logprob, cache)
        tokens.append(token)
        if step + 1 < n:
            logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
