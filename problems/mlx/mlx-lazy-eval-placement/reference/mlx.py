import mlx.core as mx


def generate(model, prompt: list[int], n: int) -> tuple[list[mx.array], mx.array]:
    logits, cache = model(mx.array(prompt))  # prefill: a graph, nothing is computed yet
    logprob = mx.array(0.0)
    tokens = []
    for step in range(n):
        last = logits[-1]
        logprobs = last - mx.logsumexp(last)
        token = mx.argmax(logprobs)
        logprob = logprob + logprobs[token]
        # The single evaluation of this step, covering everything that outlives it. The
        # cache is an ancestor of the token and would be computed anyway; the running
        # log-probability is not, and would grow by a few operations per step.
        mx.eval(token, logprob, cache)
        tokens.append(token)
        if step + 1 < n:  # the last token is returned, never fed back
            logits, cache = model(token.reshape(1), cache)
    return tokens, logprob
