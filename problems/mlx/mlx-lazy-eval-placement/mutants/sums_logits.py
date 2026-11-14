# BUG: accumulates the raw logit of each chosen token instead of its log-probability (the logsumexp normalization is missing).
import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    logits, cache = model(mx.array(prompt))
    logprob = mx.array(0.0)
    tokens = []
    for step in range(n):
        last = logits[-1]
        token = mx.argmax(last)
        logprob = logprob + last[token]
        mx.eval(token, logprob, cache)
        tokens.append(token)
        if step + 1 < n:
            logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
