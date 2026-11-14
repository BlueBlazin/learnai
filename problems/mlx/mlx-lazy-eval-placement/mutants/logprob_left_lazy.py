# BUG: each step evaluates the token and the cache but not the running log-probability, which is not an ancestor of the token; it is returned as an unevaluated chain through every step.
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
        mx.eval(token, cache)
        tokens.append(token)
        if step + 1 < n:
            logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
