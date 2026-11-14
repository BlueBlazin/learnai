# BUG: evaluates the token and then, in a second mx.eval, the log-probability and the cache: two evaluations per generated token instead of one.
import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    logits, cache = model(mx.array(prompt))
    logprob = mx.array(0.0)
    tokens = []
    for step in range(n):
        last = logits[-1]
        logprobs = last - mx.logsumexp(last)
        token = mx.argmax(logprobs)
        logprob = logprob + logprobs[token]
        mx.eval(token)
        mx.eval(logprob, cache)
        tokens.append(token)
        if step + 1 < n:
            logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
