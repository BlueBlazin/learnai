# BUG: feeds the last generated token to the model as well, a forward call (n + 1 in all) whose logits nothing reads.
import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    logits, cache = model(mx.array(prompt))
    logprob = mx.array(0.0)
    tokens = []
    for _ in range(n):
        last = logits[-1]
        logprobs = last - mx.logsumexp(last)
        token = mx.argmax(logprobs)
        logprob = logprob + logprobs[token]
        mx.eval(token, logprob, cache)
        tokens.append(token)
        logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
