# BUG: nothing is evaluated inside the loop; one mx.eval after it computes a graph that holds every step, so each model call is handed an ever larger unevaluated graph.
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
        tokens.append(token)
        if step + 1 < n:
            logits, cache = model(token.reshape(1), cache)
    mx.eval(tokens, logprob)
    return tokens, logprob
